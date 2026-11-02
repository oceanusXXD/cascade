from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
from script.model_usage import estimate_prompt_token_breakdown
from script.pair_text import format_pair_embedding_text as _format_pair_embedding_text
from script.text import split_into_sentences
from script.task_schema import resolve_embedding_payload
from script.vector import normalize_vectors
from script.vllm import get_local_vllm_gpu_profile
PAIR_EMBEDDING_VERSION = "stage_query_document_v2"
_PAIR_EMBED_REQUEST_BATCH_SIZE = 32

def pair_embed_request_batch_size() -> int:
    gpu_profile = get_local_vllm_gpu_profile() or {}
    profile_default = max(
        1,
        int(gpu_profile.get("embed_request_batch_size", _PAIR_EMBED_REQUEST_BATCH_SIZE) or _PAIR_EMBED_REQUEST_BATCH_SIZE),
    )
    try:
        return max(1, int(str(os.environ.get("CASCADE_PAIR_EMBED_REQUEST_BATCH_SIZE", profile_default)).strip()))
    except (TypeError, ValueError):
        return profile_default

def pair_embedding_source_text(mode: str, row: dict[str, Any], *, stage: str = "stage1") -> str:
    normalized_mode = str(mode or "").strip().lower()
    normalized_stage = str(stage or "").strip().lower() or "stage1"
    if normalized_stage not in {"stage1", "stage2"}:
        raise ValueError(f"unsupported pair embedding stage: {stage!r}")
    if normalized_stage == "stage2":
        return str(row.get("document_text", row.get("evidence_text", "")) or "")
    if normalized_mode == "document":
        return str(row.get("document_text", "") or "")
    if normalized_mode == "chunk":
        return str(row.get("evidence_text", "") or "")
    raise ValueError(f"unsupported algorithm mode for pair embeddings: {mode!r}")

def chunk_text_for_embedding(text: str, *, target_chars: int = 3000, overlap_chars: int = 300) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        normalized = str(text or "").strip()
        return [normalized] if normalized else [""]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len > target_chars:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            overlap_text = chunk[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)
        current.append(sentence)
        current_len += sentence_len + 1
    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [str(text or "").strip()]

def mean_pool_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    pooled = np.mean(vectors, axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(pooled))
    if norm <= 1e-12:
        return pooled.astype(np.float32)
    return (pooled / norm).astype(np.float32)

def _pair_embedding_prompt_tokens(
    *,
    chunk_text: str,
    payload: dict[str, str],
) -> dict[str, int]:
    return estimate_prompt_token_breakdown(
        system_texts=[
            payload["task_label"],
            payload["evidence_label"],
            payload["instruction_text"],
        ],
        user_texts=[payload["task_text"], chunk_text],
    )

def _formatted_pair_inputs_for_row(
    mode: str,
    row: dict[str, Any],
    *,
    stage: str = "stage1",
) -> list[dict[str, Any]]:
    source_text = pair_embedding_source_text(mode, row, stage=stage)
    payload_source = dict(row)
    payload_source["embedding_evidence_text"] = source_text
    payload = resolve_embedding_payload(
        payload_source,
        default_task_text=str(row.get("query_text", "") or ""),
        default_evidence_text=source_text,
    )
    source_chunks = chunk_text_for_embedding(payload["evidence_text"])
    inputs: list[dict[str, Any]] = []
    for chunk in source_chunks:
        token_counts = _pair_embedding_prompt_tokens(chunk_text=chunk, payload=payload)
        inputs.append(
            {
                "text": _format_pair_embedding_text(
                    chunk,
                    payload["task_text"],
                    task_label=payload["task_label"],
                    evidence_label=payload["evidence_label"],
                    instruction_text=payload["instruction_text"],
                ),
                **token_counts,
            }
        )
    return inputs

def write_evidence_row_embeddings_npy(
    mode: str,
    rows: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    stage: str = "stage1",
    base_url: str,
    model: str,
    embed_texts_func: Callable[..., np.ndarray],
    usage_callback: Callable[[dict[str, int]], None] | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> tuple[int, int]:
    total_rows = int(len(rows))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if total_rows <= 0:
        np.save(output, np.zeros((0, 0), dtype=np.float32))
        if usage_callback is not None:
            usage_callback(
                {
                    "call_count": 0,
                    "request_count": 0,
                    "input_count": 0,
                    "prompt_tokens": 0,
                    "system_prompt_tokens": 0,
                    "user_prompt_tokens": 0,
                }
            )
        if progress_callback is not None:
            progress_callback({"completed": 0, "total": 0, "request_count": 0, "input_count": 0})
        return 0, 0
    request_batch_size = pair_embed_request_batch_size()
    pending_texts: list[str] = []
    pending_spans: list[tuple[int, int]] = []
    row_cursor = 0
    memmap: np.memmap | None = None
    dimension: int | None = None
    request_count = 0
    input_count = 0
    prompt_tokens = 0
    system_prompt_tokens = 0
    user_prompt_tokens = 0
    pending_system_prompt_tokens = 0
    pending_user_prompt_tokens = 0
    def flush_pending() -> None:
        nonlocal pending_texts, pending_spans, row_cursor, memmap, dimension
        nonlocal request_count, input_count, prompt_tokens, system_prompt_tokens, user_prompt_tokens
        nonlocal pending_system_prompt_tokens, pending_user_prompt_tokens
        if not pending_texts:
            return
        batch_texts = list(pending_texts)
        batch_spans = list(pending_spans)
        pending_texts = []
        pending_spans = []
        request_count += int((len(batch_texts) + request_batch_size - 1) // request_batch_size)
        input_count += int(len(batch_texts))
        system_prompt_tokens += int(pending_system_prompt_tokens)
        user_prompt_tokens += int(pending_user_prompt_tokens)
        prompt_tokens += int(pending_system_prompt_tokens + pending_user_prompt_tokens)
        pending_system_prompt_tokens = 0
        pending_user_prompt_tokens = 0
        flat_vectors = embed_texts_func(
            base_url=base_url,
            model=model,
            texts=batch_texts,
            request_batch_size=request_batch_size,
            timeout=180,
        )
        flat_vectors = np.asarray(flat_vectors, dtype=np.float32)
        if flat_vectors.ndim != 2 or flat_vectors.shape[0] != len(batch_texts):
            raise RuntimeError(
                "embedding response size mismatch while streaming pair embeddings: "
                f"expected={len(batch_texts)} actual={flat_vectors.shape[0] if flat_vectors.ndim >= 1 else 0}"
            )
        if flat_vectors.shape[1] <= 0:
            raise RuntimeError("embedding backend returned zero-dimensional vectors")
        pooled_rows: list[np.ndarray] = []
        for start, end in batch_spans:
            pooled_rows.append(mean_pool_vectors(flat_vectors[start:end]))
        batch_rows = normalize_vectors(np.vstack(pooled_rows).astype(np.float32))
        if dimension is None:
            dimension = int(batch_rows.shape[1])
            memmap = np.lib.format.open_memmap(
                output,
                mode="w+",
                dtype=np.float32,
                shape=(total_rows, dimension),
            )
        elif int(batch_rows.shape[1]) != int(dimension):
            raise RuntimeError(
                f"embedding dimension changed while streaming pair embeddings: "
                f"expected={dimension} actual={batch_rows.shape[1]}"
            )
        assert memmap is not None
        end_row = row_cursor + int(batch_rows.shape[0])
        if end_row > total_rows:
            raise RuntimeError("streamed pair embedding row count exceeded evidence row count")
        memmap[row_cursor:end_row, :] = batch_rows
        row_cursor = end_row
        if progress_callback is not None:
            progress_callback(
                {
                    "completed": int(row_cursor),
                    "total": int(total_rows),
                    "request_count": int(request_count),
                    "input_count": int(input_count),
                }
            )
    cursor = 0
    for row in rows:
        formatted_inputs = _formatted_pair_inputs_for_row(mode, row, stage=stage)
        formatted = [str(item["text"]) for item in formatted_inputs]
        if pending_texts and len(pending_texts) + len(formatted) > request_batch_size:
            flush_pending()
            cursor = 0
        start = cursor
        pending_texts.extend(formatted)
        pending_system_prompt_tokens += sum(
            int(item.get("system_prompt_tokens", 0) or 0) for item in formatted_inputs
        )
        pending_user_prompt_tokens += sum(
            int(item.get("user_prompt_tokens", 0) or 0) for item in formatted_inputs
        )
        cursor += len(formatted)
        pending_spans.append((start, cursor))
    flush_pending()
    if row_cursor != total_rows:
        raise RuntimeError(
            f"streamed pair embedding row count mismatch: expected={total_rows} actual={row_cursor}"
        )
    if memmap is not None:
        memmap.flush()
        del memmap
    if usage_callback is not None:
        usage_callback(
            {
                "call_count": 1,
                "request_count": int(request_count),
                "input_count": int(input_count),
                "prompt_tokens": int(prompt_tokens),
                "system_prompt_tokens": int(system_prompt_tokens),
                "user_prompt_tokens": int(user_prompt_tokens),
            }
        )
    return int(total_rows), int(dimension or 0)

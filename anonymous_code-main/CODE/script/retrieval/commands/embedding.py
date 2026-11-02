from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
from script.retrieval.algorithms.embedding import (
    EMBEDDING_CANDIDATE_STRATEGY,
    build_embedding_candidate_row,
    embed_bm25_rows,
    embedding_text_batch_size,
    embedding_text_count_for_bm25_row,
)
from script.retrieval.utils.artifacts import (
    bm25_candidates_meta_path,
    bm25_candidates_path,
    embedding_candidates_meta_path,
)
from script.retrieval.utils.usage import build_embedding_model_usage
from script.retrieval.settings import ChunkSelectionSettings, DEFAULT_CHUNK_SELECTOR_VERSION
from script.io import count_jsonl_rows, iter_jsonl_objects, read_optional_json_object, remove_existing_files, write_json
from script.model_usage import build_cost_summary, elapsed_seconds_since, monotonic_seconds
from script.stream import StreamingRunWriter
from script.text import SENTENCE_TOKENIZER_NAME
from script.vllm import release_local_vllm_embedding_runner_cache, vllm_embed_texts

def _reset_embedding_outputs(root: Path) -> None:
    remove_existing_files(
        root,
        (
            "embedding_candidates.jsonl",
            "embedding_candidates.meta.json",
            "retrieval_candidates.jsonl",
            "retrieval_candidates.meta.json",
            "events.jsonl",
            "progress.json",
            "summary.json",
        ),
    )

def _load_bm25_meta(root: str | Path) -> dict[str, Any]:
    return read_optional_json_object(bm25_candidates_meta_path(root))

def build_embedding_candidates(
    settings: ChunkSelectionSettings,
    *,
    bm25_root: str | Path,
    embed_texts_func: Callable[..., np.ndarray] = vllm_embed_texts,
) -> dict[str, object]:
    started_at = monotonic_seconds()
    bm25_file = bm25_candidates_path(bm25_root)
    if not bm25_file.exists():
        raise FileNotFoundError(f"bm25 candidates file not found: {bm25_file}")
    if not str(settings.embedding_model or "").strip():
        raise RuntimeError("retrieval embedding requires --embedding-model")
    root = Path(settings.artifacts_root)
    _reset_embedding_outputs(root)
    writer = StreamingRunWriter(root)
    writer.write_event(
        "embedding_started",
        stage="embedding",
        strategy=EMBEDDING_CANDIDATE_STRATEGY,
        source_bm25_candidates_file=str(bm25_file),
    )
    total = count_jsonl_rows(bm25_file)
    completed_rows = 0
    embedding_candidate_count = 0
    totals = {
        "embedding_model_call_count": 0,
        "embedding_model_request_count": 0,
        "embedding_model_input_count": 0,
        "embedding_prompt_tokens": 0,
        "embedding_system_prompt_tokens": 0,
        "embedding_user_prompt_tokens": 0,
    }
    target_text_batch_size = embedding_text_batch_size()
    pending_rows: list[dict[str, Any]] = []
    pending_text_count = 0
    def flush_pending_rows() -> None:
        nonlocal completed_rows, embedding_candidate_count, pending_text_count
        if not pending_rows:
            return
        embedded_rows = embed_bm25_rows(
            pending_rows,
            settings=settings,
            embed_texts_func=embed_texts_func,
        )
        for embedded in embedded_rows:
            totals["embedding_model_call_count"] += int(embedded.usage.call_count)
            totals["embedding_model_request_count"] += int(embedded.usage.request_count)
            totals["embedding_model_input_count"] += int(embedded.usage.input_count)
            totals["embedding_prompt_tokens"] += int(embedded.usage.prompt_tokens)
            totals["embedding_system_prompt_tokens"] += int(embedded.usage.system_prompt_tokens)
            totals["embedding_user_prompt_tokens"] += int(embedded.usage.user_prompt_tokens)
            output_row = build_embedding_candidate_row(
                embedded,
                settings=settings,
                selector_version=DEFAULT_CHUNK_SELECTOR_VERSION,
            )
            completed_rows += 1
            embedding_candidate_count += int(len(output_row.get("chunks", []) or output_row.get("topk_chunks", []) or []))
            writer.write_stage_row("embedding_candidates", output_row)
            writer.write_progress(
                stage="embedding",
                completed=completed_rows,
                total=total,
                strategy=EMBEDDING_CANDIDATE_STRATEGY,
            )
        pending_rows.clear()
        pending_text_count = 0
    try:
        for row in iter_jsonl_objects(bm25_file):
            row_text_count = max(1, int(embedding_text_count_for_bm25_row(row)))
            if pending_rows and pending_text_count + row_text_count > target_text_batch_size:
                flush_pending_rows()
            pending_rows.append(dict(row))
            pending_text_count += row_text_count
            if pending_text_count >= target_text_batch_size:
                flush_pending_rows()
        flush_pending_rows()
    finally:
        release_local_vllm_embedding_runner_cache()
    model_usage = build_embedding_model_usage(settings, totals)
    bm25_meta = _load_bm25_meta(bm25_root)
    elapsed_seconds = elapsed_seconds_since(started_at)
    cost = build_cost_summary(elapsed_seconds=elapsed_seconds, model_usage=model_usage)
    meta = {
        "strategy": EMBEDDING_CANDIDATE_STRATEGY,
        "selector_version": DEFAULT_CHUNK_SELECTOR_VERSION,
        "sentence_tokenizer": SENTENCE_TOKENIZER_NAME,
        "source_bm25_candidates_file": str(bm25_file),
        "source_bm25_candidates_meta": bm25_meta,
        "row_count": int(completed_rows),
        "window_size": int(bm25_meta.get("window_size", settings.window_size) or settings.window_size),
        "rrf_k": int(bm25_meta.get("rrf_k", settings.rrf_k) or settings.rrf_k),
        "embedding_candidate_count": int(embedding_candidate_count),
        "embedding_text_batch_size": int(target_text_batch_size),
        **totals,
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    write_json(embedding_candidates_meta_path(root), meta)
    writer.write_event(
        "embedding_finished",
        stage="embedding",
        strategy=EMBEDDING_CANDIDATE_STRATEGY,
        row_count=completed_rows,
    )
    summary = {
        "strategy": EMBEDDING_CANDIDATE_STRATEGY,
        "row_count": int(completed_rows),
        "artifacts_root": str(root),
        "source_bm25_candidates_file": str(bm25_file),
        "embedding_candidate_count": int(embedding_candidate_count),
        "embedding_text_batch_size": int(target_text_batch_size),
        **totals,
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    writer.write_summary(summary)
    return summary
__all__ = ["build_embedding_candidates"]

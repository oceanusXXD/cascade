from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
from typing import Any
import numpy as np
from script.retrieval.settings import ChunkSelectionSettings
from script.retrieval.utils.artifacts import chunk_list, validate_embedding_candidate_row
from script.retrieval.utils.numeric import safe_float, safe_int
from script.retrieval.utils.rows import StrategySample, build_strategy_sample
from script.retrieval.utils.text import effective_query_text, vector_queries
from script.model_usage import estimate_texts_tokens
from script.vector import normalize_vectors
from script.vllm_common import get_local_vllm_gpu_profile
DEFAULT_EMBED_REQUEST_BATCH_SIZE = 32
EMBEDDING_CANDIDATE_STRATEGY = "bm25_embedding_scores"

@dataclass(frozen=True)

class EmbeddingUsage:
    call_count: int
    request_count: int
    input_count: int
    prompt_tokens: int
    system_prompt_tokens: int
    user_prompt_tokens: int

@dataclass(frozen=True)

class EmbeddingScoringResult:
    vectors: np.ndarray
    usage: EmbeddingUsage

@dataclass(frozen=True)

class EmbeddedBm25Row:
    source_row: Mapping[str, Any]
    sample: StrategySample
    chunks: list[dict[str, Any]]
    vector_scores: list[float]
    usage: EmbeddingUsage

def embedding_request_batch_size() -> int:
    gpu_profile = get_local_vllm_gpu_profile() or {}
    profile_default = max(
        1,
        int(
            gpu_profile.get("embed_request_batch_size", DEFAULT_EMBED_REQUEST_BATCH_SIZE)
            or DEFAULT_EMBED_REQUEST_BATCH_SIZE
        ),
    )
    try:
        return max(
            1,
            int(
                str(
                    os.environ.get(
                        "CASCADE_BATCHPROCESS_EMBED_REQUEST_BATCH_SIZE",
                        profile_default,
                    )
                ).strip()
            ),
        )
    except (TypeError, ValueError):
        return profile_default

def embedding_text_batch_size() -> int:
    gpu_profile = get_local_vllm_gpu_profile() or {}
    profile_default = max(
        DEFAULT_EMBED_REQUEST_BATCH_SIZE,
        int(gpu_profile.get("embed_request_batch_size", DEFAULT_EMBED_REQUEST_BATCH_SIZE) or 0),
        int(gpu_profile.get("embed_max_num_seqs", 0) or 0),
    )
    try:
        return max(
            1,
            int(
                str(
                    os.environ.get(
                        "CASCADE_BATCHPROCESS_EMBED_TEXT_BATCH_SIZE",
                        profile_default,
                    )
                ).strip()
            ),
        )
    except (TypeError, ValueError):
        return max(1, int(profile_default))

def _empty_usage() -> EmbeddingUsage:
    return EmbeddingUsage(
        call_count=0,
        request_count=0,
        input_count=0,
        prompt_tokens=0,
        system_prompt_tokens=0,
        user_prompt_tokens=0,
    )

def embed_texts_with_usage(
    *,
    settings: ChunkSelectionSettings,
    texts: Sequence[str],
    embed_texts_func: Callable[..., np.ndarray],
) -> EmbeddingScoringResult:
    items = [str(text or "") for text in texts]
    if not items:
        return EmbeddingScoringResult(vectors=np.zeros((0, 0), dtype=np.float32), usage=_empty_usage())
    if not str(settings.embedding_model or "").strip():
        return EmbeddingScoringResult(vectors=np.zeros((len(items), 0), dtype=np.float32), usage=_empty_usage())
    request_batch_size = embedding_request_batch_size()
    prompt_tokens = estimate_texts_tokens(items)
    usage = EmbeddingUsage(
        call_count=1,
        request_count=int(math.ceil(len(items) / float(request_batch_size))),
        input_count=len(items),
        prompt_tokens=prompt_tokens,
        system_prompt_tokens=0,
        user_prompt_tokens=prompt_tokens,
    )
    vectors = embed_texts_func(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        texts=items,
        api_key=str(settings.embedding_api_key or "EMPTY"),
        timeout=int(settings.embedding_timeout),
        request_batch_size=int(request_batch_size),
        normalize=True,
    )
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] != len(items):
        raise RuntimeError(
            f"embedding response size mismatch: expected={len(items)} actual={arr.shape[0]}"
        )
    return EmbeddingScoringResult(vectors=normalize_vectors(arr), usage=usage)

def max_query_candidate_scores(
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    candidate_count: int,
) -> list[float]:
    if query_vectors.size == 0 or candidate_vectors.size == 0:
        return [0.0 for _ in range(int(candidate_count))]
    similarities = np.matmul(query_vectors, candidate_vectors.T)
    return [
        float(np.max(similarities[:, idx]))
        for idx in range(int(candidate_count))
    ]

def embed_bm25_row(
    row: Mapping[str, Any],
    *,
    settings: ChunkSelectionSettings,
    embed_texts_func: Callable[..., np.ndarray],
) -> EmbeddedBm25Row:
    return embed_bm25_rows(
        [row],
        settings=settings,
        embed_texts_func=embed_texts_func,
    )[0]

def _row_embedding_inputs(row: Mapping[str, Any]) -> tuple[StrategySample, list[dict[str, Any]], list[tuple[str, str]], list[str]]:
    chunks = [dict(chunk) for chunk in chunk_list(row) if isinstance(chunk, Mapping)]
    sample = build_strategy_sample(
        row,
        query_rewrites=tuple(str(item) for item in list(row.get("query_rewrites", []) or [])),
        hyde_hypotheses=tuple(str(item) for item in list(row.get("hyde_hypotheses", []) or [])),
    )
    vector_query_items = vector_queries(sample)
    embedding_texts = [query for _, query in vector_query_items] + [
        str(chunk.get("text", "") or "") for chunk in chunks
    ]
    return sample, chunks, vector_query_items, embedding_texts

def embedding_text_count_for_bm25_row(row: Mapping[str, Any]) -> int:
    _, chunks, vector_query_items, _ = _row_embedding_inputs(row)
    return int(len(vector_query_items) + len(chunks))

def embed_bm25_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    settings: ChunkSelectionSettings,
    embed_texts_func: Callable[..., np.ndarray],
) -> list[EmbeddedBm25Row]:
    source_rows = [dict(row) for row in list(rows or [])]
    if not source_rows:
        return []
    plans: list[tuple[Mapping[str, Any], StrategySample, list[dict[str, Any]], int, int]] = []
    embedding_texts: list[str] = []
    cursor = 0
    for source_row in source_rows:
        sample, chunks, vector_query_items, row_texts = _row_embedding_inputs(source_row)
        query_count = len(vector_query_items)
        plans.append((source_row, sample, chunks, cursor, query_count))
        embedding_texts.extend(row_texts)
        cursor += len(row_texts)
    embedding_result = embed_texts_with_usage(
        settings=settings,
        texts=embedding_texts,
        embed_texts_func=embed_texts_func,
    )
    outputs: list[EmbeddedBm25Row] = []
    assigned_usage = False
    for source_row, sample, chunks, start, query_count in plans:
        end = start + query_count + len(chunks)
        row_vectors = embedding_result.vectors[start:end]
        query_vectors = row_vectors[:query_count]
        candidate_vectors = row_vectors[query_count:]
        vector_scores = max_query_candidate_scores(
            query_vectors,
            candidate_vectors,
            candidate_count=len(chunks),
        )
        usage = embedding_result.usage if not assigned_usage else _empty_usage()
        assigned_usage = True
        outputs.append(
            EmbeddedBm25Row(
                source_row=source_row,
                sample=sample,
                chunks=chunks,
                vector_scores=vector_scores,
                usage=usage,
            )
        )
    return outputs

def build_embedding_candidate_row(
    embedded: EmbeddedBm25Row,
    *,
    settings: ChunkSelectionSettings,
    selector_version: str,
) -> dict[str, Any]:
    chunks = [dict(chunk) for chunk in list(embedded.chunks or [])]
    vector_scores = [float(score) for score in list(embedded.vector_scores or [])]
    if len(chunks) != len(vector_scores):
        raise RuntimeError(
            "embedding score count mismatch: "
            f"chunks={len(chunks)} embedding={len(vector_scores)}"
        )
    chunks_out = []
    for rank, source_chunk in enumerate(chunks, start=1):
        vector_score = float(vector_scores[rank - 1])
        chunks_out.append(
            {
                **source_chunk,
                "rank": int(rank),
                "candidate_index": safe_int(source_chunk.get("candidate_index"), rank - 1),
                "sentence_indices": [
                    int(index) for index in list(source_chunk.get("sentence_indices", []) or [])
                ],
                "text": str(source_chunk.get("text", "") or ""),
                "lexical_score": safe_float(
                    source_chunk.get("lexical_score", source_chunk.get("bm25_score")),
                    0.0,
                ),
                "bm25_score": safe_float(
                    source_chunk.get("bm25_score", source_chunk.get("lexical_score")),
                    0.0,
                ),
                "vector_score": vector_score,
                "embedding_score": vector_score,
            }
        )
    source_row = embedded.source_row
    sample = embedded.sample
    effective_query = str(effective_query_text(sample))
    row = {
        "sample_id": str(sample.sample_id),
        "query_id": str(sample.query_id),
        "document_id": str(sample.document_id),
        "query_text": str(sample.query_text),
        "raw_query_text": str(sample.raw_query_text or sample.query_text),
        "claim": str(sample.claim_text or sample.query_text),
        "claim_text": str(sample.claim_text or sample.query_text),
        "prompt_style": str(sample.prompt_style or "query_document"),
        "prompt_task_label": str(sample.prompt_task_label),
        "prompt_evidence_label": str(sample.prompt_evidence_label),
        "prompt_task_text": str(sample.prompt_task_text),
        "prompt_instruction_text": str(sample.prompt_instruction_text),
        "prompt_evidence_text": str(sample.prompt_evidence_text),
        "embedding_task_text": str(sample.embedding_task_text),
        "embedding_instruction_text": str(sample.embedding_instruction_text),
        "embedding_evidence_text": str(sample.embedding_evidence_text),
        "document_text": str(sample.document_text),
        "retrieval_query_text": effective_query,
        "strategy": EMBEDDING_CANDIDATE_STRATEGY,
        "selector_version": str(selector_version),
        "label": str(sample.label or ""),
        "query_rewrites": [str(item) for item in list(sample.query_rewrites or ())],
        "hyde_hypotheses": [str(item) for item in list(sample.hyde_hypotheses or ())],
        "selected_query_variant": effective_query,
        "selected_query_source": "embedding",
        "sentence_count": safe_int(source_row.get("sentence_count"), 0),
        "retrieval_candidate_count": safe_int(source_row.get("retrieval_candidate_count"), len(chunks_out)),
        "retrieval_topk_count": int(len(chunks_out)),
        "window_size": safe_int(source_row.get("window_size"), settings.window_size),
        "rrf_k": safe_int(source_row.get("rrf_k"), settings.rrf_k),
        "embedding_model_call_count": int(embedded.usage.call_count),
        "embedding_model_request_count": int(embedded.usage.request_count),
        "embedding_model_input_count": int(embedded.usage.input_count),
        "embedding_prompt_tokens": int(embedded.usage.prompt_tokens),
        "embedding_system_prompt_tokens": int(embedded.usage.system_prompt_tokens),
        "embedding_user_prompt_tokens": int(embedded.usage.user_prompt_tokens),
        "chunks": chunks_out,
    }
    validate_embedding_candidate_row(row)
    return row

def embedded_bm25_row_from_embedding_candidate_row(row: Mapping[str, Any]) -> EmbeddedBm25Row:
    candidate = dict(row)
    validate_embedding_candidate_row(candidate)
    chunks: list[dict[str, Any]] = []
    vector_scores: list[float] = []
    for chunk in chunk_list(candidate):
        if not isinstance(chunk, Mapping):
            continue
        normalized = dict(chunk)
        vector_score = safe_float(
            normalized.get("vector_score", normalized.get("embedding_score")),
            0.0,
        )
        normalized["vector_score"] = vector_score
        normalized["embedding_score"] = vector_score
        chunks.append(normalized)
        vector_scores.append(float(vector_score))
    return EmbeddedBm25Row(
        source_row=candidate,
        sample=build_strategy_sample(
            candidate,
            query_rewrites=tuple(str(item) for item in list(candidate.get("query_rewrites", []) or [])),
            hyde_hypotheses=tuple(str(item) for item in list(candidate.get("hyde_hypotheses", []) or [])),
        ),
        chunks=chunks,
        vector_scores=vector_scores,
        usage=EmbeddingUsage(
            call_count=safe_int(candidate.get("embedding_model_call_count"), 0),
            request_count=safe_int(candidate.get("embedding_model_request_count"), 0),
            input_count=safe_int(candidate.get("embedding_model_input_count"), 0),
            prompt_tokens=safe_int(candidate.get("embedding_prompt_tokens"), 0),
            system_prompt_tokens=safe_int(candidate.get("embedding_system_prompt_tokens"), 0),
            user_prompt_tokens=safe_int(candidate.get("embedding_user_prompt_tokens"), 0),
        ),
    )
__all__ = [
    "DEFAULT_EMBED_REQUEST_BATCH_SIZE",
    "EMBEDDING_CANDIDATE_STRATEGY",
    "EmbeddedBm25Row",
    "EmbeddingScoringResult",
    "EmbeddingUsage",
    "build_embedding_candidate_row",
    "embed_bm25_row",
    "embed_bm25_rows",
    "embed_texts_with_usage",
    "embedded_bm25_row_from_embedding_candidate_row",
    "embedding_request_batch_size",
    "embedding_text_batch_size",
    "embedding_text_count_for_bm25_row",
    "max_query_candidate_scores",
]

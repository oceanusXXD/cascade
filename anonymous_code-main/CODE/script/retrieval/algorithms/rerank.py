from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any
from script.retrieval.settings import ChunkSelectionSettings
from script.retrieval.utils.artifacts import chunk_list, validate_rerank_result_row
from script.retrieval.utils.numeric import safe_float, safe_int
from script.model_usage import estimate_prompt_token_breakdown
RERANK_CACHE_VERSION = "rerank_score_cache_v1"
RERANK_CACHE_STRATEGY = "rerank_cache"

@dataclass(frozen=True)

class RerankRowJob:
    row: Mapping[str, Any]
    rerank_query_text: str
    rerank_query_source: str
    pairs: list[tuple[str, str]]

@dataclass(frozen=True)

class RerankScoringResult:
    scores: list[float]
    call_count: int
    request_count: int
    input_count: int
    prompt_tokens: int
    system_prompt_tokens: int
    user_prompt_tokens: int
    completion_tokens: int

def _empty_result() -> RerankScoringResult:
    return RerankScoringResult(
        scores=[],
        call_count=0,
        request_count=0,
        input_count=0,
        prompt_tokens=0,
        system_prompt_tokens=0,
        user_prompt_tokens=0,
        completion_tokens=0,
    )

def _rerank_prompt_token_counts(
    *,
    instruction: Any,
    query: Any,
    document: Any,
) -> dict[str, int]:
    return estimate_prompt_token_breakdown(
        system_texts=[instruction],
        user_texts=[query, document],
    )

def score_query_document_pairs_with_usage(
    *,
    pairs: Sequence[tuple[str, str]],
    settings: ChunkSelectionSettings,
    rerank_pair_score_func: Callable[..., Sequence[float]],
    release_cache_func: Callable[[], None] | None = None,
) -> RerankScoringResult:
    items = [(str(query or ""), str(document or "")) for query, document in list(pairs or [])]
    if not items:
        return _empty_result()
    call_count = 1
    request_count = int(
        math.ceil(len(items) / float(max(1, int(settings.rerank_request_batch_size))))
    )
    system_prompt_tokens = 0
    user_prompt_tokens = 0
    for query, document in items:
        token_counts = _rerank_prompt_token_counts(
            instruction=settings.reranker_instruction,
            query=query,
            document=document,
        )
        system_prompt_tokens += int(token_counts["system_prompt_tokens"])
        user_prompt_tokens += int(token_counts["user_prompt_tokens"])
    prompt_tokens = int(system_prompt_tokens + user_prompt_tokens)
    try:
        raw_scores = rerank_pair_score_func(
            query_document_pairs=items,
            instruction=str(settings.reranker_instruction),
            model=str(settings.reranker_model),
            base_url=str(settings.reranker_base_url),
            api_key=str(settings.reranker_api_key),
            timeout=int(settings.reranker_timeout),
            request_batch_size=int(settings.rerank_request_batch_size),
            max_length=int(settings.reranker_max_length),
        )
    finally:
        if release_cache_func is not None:
            release_cache_func()
    scores = [float(score) for score in list(raw_scores or [])]
    if len(scores) != len(items):
        raise RuntimeError(
            f"reranker score count mismatch: expected={len(items)} actual={len(scores)}"
        )
    return RerankScoringResult(
        scores=scores,
        call_count=int(call_count),
        request_count=int(request_count),
        input_count=len(items),
        prompt_tokens=int(prompt_tokens),
        system_prompt_tokens=int(system_prompt_tokens),
        user_prompt_tokens=int(user_prompt_tokens),
        completion_tokens=len(items),
    )

def resolve_rerank_query(row: Mapping[str, Any]) -> tuple[str, str]:
    retrieval_query_text = str(row.get("retrieval_query_text", "") or "").strip()
    if retrieval_query_text:
        return retrieval_query_text, "retrieval_query_text"
    query_text = str(row.get("query_text", "") or "").strip()
    claim_text = str(row.get("claim_text", "") or row.get("claim", "") or "").strip()
    if claim_text:
        return f"Query:\n{query_text}\n\nClaim:\n{claim_text}", "query_claim"
    return query_text, "query_text"

def build_rerank_row_job(row: Mapping[str, Any]) -> RerankRowJob:
    rerank_query_text, rerank_query_source = resolve_rerank_query(row)
    pairs = [
        (rerank_query_text, str(chunk.get("text", "") if isinstance(chunk, Mapping) else chunk or ""))
        for chunk in chunk_list(row)
    ]
    return RerankRowJob(
        row=row,
        rerank_query_text=rerank_query_text,
        rerank_query_source=rerank_query_source,
        pairs=pairs,
    )

def _normalize_chunk(chunk: Mapping[str, Any], *, rank: int, rerank_score: float) -> dict[str, Any]:
    normalized = dict(chunk)
    normalized["rank"] = safe_int(normalized.get("rank", rank), rank)
    normalized["candidate_index"] = safe_int(normalized.get("candidate_index", 0), 0)
    normalized["sentence_indices"] = [
        safe_int(index, 0)
        for index in list(normalized.get("sentence_indices", []) or [])
    ]
    normalized["text"] = str(normalized.get("text", "") or "")
    normalized["rerank_score"] = float(rerank_score)
    return normalized

def build_rerank_result_row(
    row: Mapping[str, Any],
    *,
    rerank_query_text: str,
    rerank_query_source: str,
    scores: Sequence[float],
    settings: ChunkSelectionSettings,
    scoring_result: RerankScoringResult,
    assign_model_call_usage: bool,
) -> dict[str, Any]:
    normalized_scores = [safe_float(score) for score in scores]
    chunks_out = [
        _normalize_chunk(
            chunk if isinstance(chunk, Mapping) else {"text": str(chunk or "")},
            rank=index + 1,
            rerank_score=float(normalized_scores[index]),
        )
        for index, chunk in enumerate(chunk_list(row))
    ]
    rerank_input_count = len(chunks_out)
    system_prompt_tokens = 0
    user_prompt_tokens = 0
    for chunk in chunks_out:
        token_counts = _rerank_prompt_token_counts(
            instruction=settings.reranker_instruction,
            query=rerank_query_text,
            document=chunk.get("text", ""),
        )
        system_prompt_tokens += int(token_counts["system_prompt_tokens"])
        user_prompt_tokens += int(token_counts["user_prompt_tokens"])
    prompt_tokens = int(system_prompt_tokens + user_prompt_tokens)
    result = {
        "sample_id": str(row.get("sample_id", "") or ""),
        "query_id": str(row.get("query_id", "") or ""),
        "document_id": str(row.get("document_id", "") or ""),
        "query_text": str(row.get("query_text", "") or ""),
        "raw_query_text": str(row.get("raw_query_text", "") or row.get("query_text", "") or ""),
        "claim": str(row.get("claim", "") or row.get("claim_text", "") or row.get("query_text", "") or ""),
        "claim_text": str(row.get("claim_text", "") or row.get("claim", "") or row.get("query_text", "") or ""),
        "document_text": str(row.get("document_text", "") or ""),
        "label": str(row.get("label", "") or ""),
        "retrieval_query_text": str(row.get("retrieval_query_text", "") or rerank_query_text),
        "rerank_query_text": str(rerank_query_text),
        "rerank_query_source": str(rerank_query_source),
        "strategy": RERANK_CACHE_STRATEGY,
        "source_strategy": str(row.get("strategy", "") or ""),
        "source_selector_version": str(row.get("selector_version", "") or ""),
        "rerank_cache_version": RERANK_CACHE_VERSION,
        "sentence_count": safe_int(row.get("sentence_count", 0), 0),
        "retrieval_candidate_count": safe_int(row.get("retrieval_candidate_count", 0), 0),
        "retrieval_topk_count": safe_int(row.get("retrieval_topk_count", len(chunks_out)), len(chunks_out)),
        "rerank_candidate_count": int(rerank_input_count),
        "window_size": safe_int(row.get("window_size", 0), 0),
        "rrf_k": safe_int(row.get("rrf_k", 0), 0),
        "rerank_scores": normalized_scores,
        "chunks": chunks_out,
        "rerank_model_call_count": int(scoring_result.call_count) if assign_model_call_usage else 0,
        "rerank_model_request_count": int(scoring_result.request_count) if assign_model_call_usage else 0,
        "rerank_model_input_count": int(rerank_input_count),
        "rerank_prompt_tokens": int(prompt_tokens),
        "rerank_system_prompt_tokens": int(system_prompt_tokens),
        "rerank_user_prompt_tokens": int(user_prompt_tokens),
        "rerank_completion_tokens": int(rerank_input_count),
    }
    validate_rerank_result_row(result)
    return result
__all__ = [
    "RERANK_CACHE_STRATEGY",
    "RERANK_CACHE_VERSION",
    "RerankRowJob",
    "RerankScoringResult",
    "build_rerank_result_row",
    "build_rerank_row_job",
    "resolve_rerank_query",
    "score_query_document_pairs_with_usage",
]

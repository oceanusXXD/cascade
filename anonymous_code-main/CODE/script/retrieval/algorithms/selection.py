from __future__ import annotations
from collections.abc import Mapping, Sequence
import math
from typing import Any
from script.retrieval.algorithms.rrf import first_order_elbow_prefix_count
from script.retrieval.utils.artifacts import chunk_list
from script.retrieval.utils.numeric import safe_float, safe_int
from script.retrieval.utils.text import ChunkCandidate
from script.text import split_into_sentences
RERANK_SELECT_MIN = 1
RERANK_SELECT_MAX = 5

def rerank_selection_limit(rerank_candidate_count: int) -> int:
    total = int(rerank_candidate_count)
    if total <= 0:
        return 0
    return min(total, max(RERANK_SELECT_MIN, RERANK_SELECT_MAX))

def safe_rerank_score(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")

def rerank_elbow_prefix_count(sorted_scores: Sequence[float]) -> int:
    # Paper rule: keep the prefix ending at the largest reranker-score gap,
    # with no extra absolute/relative cliff gate.
    return first_order_elbow_prefix_count(sorted_scores)

def merge_selected_chunks(
    sentences: Sequence[str],
    selected_candidates: Sequence[ChunkCandidate],
) -> ChunkCandidate:
    selected_indices: list[int] = []
    seen: set[int] = set()
    for candidate in selected_candidates:
        for index in candidate.sentence_indices:
            resolved_index = int(index)
            if resolved_index in seen:
                continue
            seen.add(resolved_index)
            selected_indices.append(resolved_index)
    selected_indices.sort()
    merged_text = " ".join(
        str(sentences[index]).strip()
        for index in selected_indices
        if 0 <= int(index) < len(sentences) and str(sentences[index]).strip()
    ).strip()
    return ChunkCandidate(text=merged_text, sentence_indices=tuple(selected_indices))

def select_rerank_hit_indices(
    scores: Sequence[float],
    *,
    candidates: Sequence[ChunkCandidate] | None = None,
    sentences: Sequence[str] | None = None,
    token_budget: int | None = None,
) -> list[int]:
    # Retain these keyword arguments for compatibility, but do not let them
    # alter the paper's largest-gap + cap-5 selection rule.
    del candidates, sentences, token_budget
    normalized_scores = [
        safe_rerank_score(score)
        for score in list(scores or [])
    ]
    if not normalized_scores:
        return []
    ordered = [
        int(index)
        for index in sorted(
            range(len(normalized_scores)),
            key=lambda idx: (-float(normalized_scores[idx]), int(idx)),
        )
        if math.isfinite(float(normalized_scores[index]))
    ]
    if not ordered:
        return [0]
    limit = rerank_selection_limit(len(normalized_scores))
    sorted_scores = [float(normalized_scores[index]) for index in ordered]
    elbow_count = rerank_elbow_prefix_count(sorted_scores)
    prefix_count = min(int(limit), max(1, int(elbow_count)))
    return [int(index) for index in ordered[:prefix_count]] or [int(ordered[0])]

def _chunk_score(chunk: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in chunk:
            return safe_float(chunk.get(name), 0.0)
    return 0.0

def _chunk_candidate(chunk: Mapping[str, Any]) -> ChunkCandidate:
    return ChunkCandidate(
        text=str(chunk.get("text", "") or ""),
        sentence_indices=tuple(safe_int(index, 0) for index in list(chunk.get("sentence_indices", []) or [])),
    )

def _scores_for_chunks(row: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]]) -> list[float]:
    raw_scores = list(row.get("rerank_scores", []) or [])
    if raw_scores:
        scores = [safe_float(score) for score in raw_scores]
    else:
        scores = [_chunk_score(chunk, "rerank_score") for chunk in chunks]
    if len(scores) != len(chunks):
        raise RuntimeError(
            f"cached rerank score count mismatch for sample_id={row.get('sample_id', '')}: "
            f"scores={len(scores)} chunks={len(chunks)}"
        )
    return scores

def cached_rerank_selection_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    chunks = [
        dict(chunk)
        for chunk in chunk_list(row)
        if isinstance(chunk, Mapping)
    ]
    scores = _scores_for_chunks(row, chunks)
    candidates = [_chunk_candidate(chunk) for chunk in chunks]
    document_text = str(row.get("document_text", "") or "")
    sentences = split_into_sentences(document_text)
    if not sentences and document_text.strip():
        sentences = [document_text.strip()]
    selected_rerank_indices = select_rerank_hit_indices(
        scores,
        candidates=candidates,
        sentences=sentences,
    )
    if not selected_rerank_indices and candidates:
        selected_rerank_indices = [0]
    best_index = max(
        selected_rerank_indices,
        key=lambda idx: (
            safe_float(scores[int(idx)] if 0 <= int(idx) < len(scores) else 0.0),
            -int(idx),
        ),
    )
    selected_candidates = [
        candidates[int(index)]
        for index in selected_rerank_indices
        if 0 <= int(index) < len(candidates)
    ]
    merged = merge_selected_chunks(sentences, selected_candidates)
    if not str(merged.text or "").strip():
        merged = ChunkCandidate(
            text=" ".join(candidate.text for candidate in selected_candidates if str(candidate.text or "").strip()).strip(),
            sentence_indices=tuple(
                sorted(
                    {
                        int(sentence_index)
                        for candidate in selected_candidates
                        for sentence_index in candidate.sentence_indices
                    }
                )
            ),
        )
    best_chunk = dict(chunks[int(best_index)]) if 0 <= int(best_index) < len(chunks) else {}
    retrieval_candidate_count = safe_int(row.get("retrieval_candidate_count", len(chunks)), len(chunks))
    rerank_candidate_count = len(chunks)
    saved_count = max(0, retrieval_candidate_count - rerank_candidate_count)
    return {
        "selected_chunk_text": str(merged.text or ""),
        "selected_chunk_indices": [int(index) for index in list(merged.sentence_indices or ())],
        "selected_query_variant": str(row.get("rerank_query_text", "") or row.get("retrieval_query_text", "") or row.get("query_text", "") or ""),
        "selected_query_source": "reranker_cached",
        "rerank_score": safe_float(scores[int(best_index)] if 0 <= int(best_index) < len(scores) else 0.0),
        "rerank_scores": [safe_float(score) for score in scores],
        "lexical_score": _chunk_score(best_chunk, "lexical_score", "bm25_score"),
        "vector_score": _chunk_score(best_chunk, "vector_score", "embedding_score"),
        "rrf_score": _chunk_score(best_chunk, "rrf_score"),
        "sentence_count": safe_int(row.get("sentence_count", len(sentences)), len(sentences)),
        "retrieval_k": 0,
        "rerank_m": safe_int(row.get("retrieval_topk_count", rerank_candidate_count), rerank_candidate_count),
        "retrieval_candidate_count": int(retrieval_candidate_count),
        "rerank_candidate_count": int(rerank_candidate_count),
        "embedding_model_call_count": 0,
        "embedding_model_request_count": 0,
        "embedding_model_input_count": 0,
        "embedding_prompt_tokens": 0,
        "rerank_model_call_count": 0,
        "rerank_model_request_count": 0,
        "rerank_model_input_count": 0,
        "rerank_prompt_tokens": 0,
        "rerank_completion_tokens": 0,
        "full_rerank_candidate_count": int(retrieval_candidate_count),
        "rerank_saved_candidate_count": int(saved_count),
        "rerank_reduction_ratio": (
            float(saved_count) / float(retrieval_candidate_count)
            if retrieval_candidate_count > 0
            else 0.0
        ),
    }
__all__ = [
    "RERANK_SELECT_MAX",
    "RERANK_SELECT_MIN",
    "cached_rerank_selection_from_row",
    "merge_selected_chunks",
    "rerank_elbow_prefix_count",
    "rerank_selection_limit",
    "safe_rerank_score",
    "select_rerank_hit_indices",
]

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any
from script.retrieval.algorithms.embedding import EmbeddedBm25Row
from script.retrieval.settings import ChunkSelectionSettings
from script.retrieval.utils.rows import build_retrieval_candidate_row
from script.retrieval.utils.text import effective_query_text
RERANK_ENTRY_ELBOW_BUFFER = 5
RRF_CANDIDATE_STRATEGY = "bm25_embedding_rrf_candidates"


@dataclass(frozen=True)

class FusedCandidateResult:
    indices: list[int]
    scores: list[float]
    candidate_count: int

def rank_positions(scores: Sequence[float]) -> dict[int, int]:
    ordered = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    return {idx: rank for rank, idx in enumerate(ordered, start=1)}

def has_score_signal(scores: Sequence[float]) -> bool:
    finite_scores = [float(score) for score in list(scores or []) if math.isfinite(float(score))]
    return bool(finite_scores) and (max(finite_scores) - min(finite_scores)) > 1e-12

def rrf_score(*ranks: int, k: int) -> float:
    score = 0.0
    for rank in ranks:
        if rank > 0:
            score += 1.0 / float(k + rank)
    return float(score)

def first_order_elbow_prefix_count(
    sorted_scores: Sequence[float],
    *,
    buffer: int = 0,
    min_absolute_drop: float | None = None,
    min_relative_drop: float | None = None,
) -> int:
    raw_scores = [] if sorted_scores is None else list(sorted_scores)
    values = [float(score) for score in raw_scores if math.isfinite(float(score))]
    if len(values) <= 1:
        return len(values)
    best_gap_index: int | None = None
    best_gap = 0.0
    for index, (left, right) in enumerate(zip(values, values[1:])):
        gap = max(0.0, float(left) - float(right))
        if gap <= best_gap:
            continue
        relative_drop = gap / max(abs(float(left)), 1e-12)
        if min_absolute_drop is not None and gap < float(min_absolute_drop):
            continue
        if min_relative_drop is not None and relative_drop < float(min_relative_drop):
            continue
        best_gap_index = int(index)
        best_gap = float(gap)
    if best_gap_index is None:
        return len(values)
    return min(len(values), int(best_gap_index) + 1 + max(0, int(buffer)))

def rerank_entry_count_from_sorted_scores(sorted_scores: Sequence[float]) -> int:
    return first_order_elbow_prefix_count(
        sorted_scores,
        buffer=RERANK_ENTRY_ELBOW_BUFFER,
    )

def fused_candidate_indices(
    *,
    lexical_scores: list[float],
    vector_scores: list[float],
    rrf_k: int,
    rerank_entry_count: int | None = None,
) -> FusedCandidateResult:
    if len(lexical_scores) != len(vector_scores):
        raise RuntimeError(
            "BM25 and embedding score count mismatch: "
            f"bm25={len(lexical_scores)} embedding={len(vector_scores)}"
        )
    all_indices = set(range(len(lexical_scores)))
    lexical_ranks = rank_positions(lexical_scores) if has_score_signal(lexical_scores) else {}
    vector_ranks = rank_positions(vector_scores) if has_score_signal(vector_scores) else {}
    fused_scores = [
        rrf_score(
            lexical_ranks.get(idx, 0),
            vector_ranks.get(idx, 0),
            k=int(rrf_k),
        )
        for idx in range(len(lexical_scores))
    ]
    ordered = sorted(
        all_indices,
        key=lambda idx: (
            -float(fused_scores[idx]),
            -float(vector_scores[idx]),
            -float(lexical_scores[idx]),
            idx,
        ),
    )
    resolved_entry_count = (
        int(rerank_entry_count)
        if rerank_entry_count is not None
        else rerank_entry_count_from_sorted_scores([float(fused_scores[index]) for index in ordered])
    )
    resolved_entry_count = max(0, min(len(ordered), int(resolved_entry_count)))
    if resolved_entry_count >= len(all_indices):
        return FusedCandidateResult(
            indices=list(range(len(lexical_scores))),
            scores=fused_scores,
            candidate_count=len(all_indices),
        )
    return FusedCandidateResult(
        indices=ordered[:resolved_entry_count],
        scores=fused_scores,
        candidate_count=len(all_indices),
    )

def _lexical_scores(chunks: list[Mapping[str, Any]]) -> list[float]:
    return [float(chunk.get("lexical_score", chunk.get("bm25_score", 0.0)) or 0.0) for chunk in chunks]

def build_rrf_candidate_row(
    embedded: EmbeddedBm25Row,
    *,
    settings: ChunkSelectionSettings,
    selector_version: str,
) -> dict[str, Any]:
    chunks = list(embedded.chunks)
    fused = fused_candidate_indices(
        lexical_scores=_lexical_scores(chunks),
        vector_scores=embedded.vector_scores,
        rrf_k=int(settings.rrf_k),
    )
    chunks_out = []
    for rank, candidate_index in enumerate(fused.indices, start=1):
        source_chunk = chunks[int(candidate_index)]
        chunks_out.append(
            {
                **source_chunk,
                "rank": int(rank),
                "candidate_index": int(source_chunk.get("candidate_index", candidate_index) or candidate_index),
                "sentence_indices": [
                    int(index) for index in list(source_chunk.get("sentence_indices", []) or [])
                ],
                "text": str(source_chunk.get("text", "") or ""),
                "lexical_score": float(source_chunk.get("lexical_score", source_chunk.get("bm25_score", 0.0)) or 0.0),
                "bm25_score": float(source_chunk.get("bm25_score", source_chunk.get("lexical_score", 0.0)) or 0.0),
                "vector_score": float(embedded.vector_scores[int(candidate_index)]),
                "embedding_score": float(embedded.vector_scores[int(candidate_index)]),
                "rrf_score": float(fused.scores[int(candidate_index)]),
            }
        )
    source_row = embedded.source_row
    sample = embedded.sample
    retrieval = {
        "retrieval_query_text": str(effective_query_text(sample)),
        "selected_query_variant": str(effective_query_text(sample)),
        "selected_query_source": "bm25_embedding_rrf",
        "sentence_count": int(source_row.get("sentence_count", 0) or 0),
        "retrieval_candidate_count": int(fused.candidate_count),
        "retrieval_topk_count": int(len(chunks_out)),
        "window_size": int(source_row.get("window_size", settings.window_size) or settings.window_size),
        "embedding_model_call_count": int(embedded.usage.call_count),
        "embedding_model_request_count": int(embedded.usage.request_count),
        "embedding_model_input_count": int(embedded.usage.input_count),
        "embedding_prompt_tokens": int(embedded.usage.prompt_tokens),
        "embedding_system_prompt_tokens": int(embedded.usage.system_prompt_tokens),
        "embedding_user_prompt_tokens": int(embedded.usage.user_prompt_tokens),
        "chunks": chunks_out,
    }
    return build_retrieval_candidate_row(
        sample,
        strategy_name=RRF_CANDIDATE_STRATEGY,
        selector_version=str(selector_version),
        retrieval=retrieval,
    )
__all__ = [
    "FusedCandidateResult",
    "RERANK_ENTRY_ELBOW_BUFFER",
    "RRF_CANDIDATE_STRATEGY",
    "build_rrf_candidate_row",
    "first_order_elbow_prefix_count",
    "fused_candidate_indices",
    "has_score_signal",
    "rank_positions",
    "rerank_entry_count_from_sorted_scores",
    "rrf_score",
]

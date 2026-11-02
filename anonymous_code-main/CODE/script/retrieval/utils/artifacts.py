from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping

def chunk_list(row: Mapping[str, Any] | dict[str, Any]) -> list[Any]:
    chunks = row.get("chunks") if isinstance(row, Mapping) else None
    if chunks is None:
        chunks = row.get("topk_chunks") if isinstance(row, Mapping) else None
    return list(chunks or [])

def has_chunk_field(row: Mapping[str, Any] | dict[str, Any]) -> bool:
    return isinstance(row, Mapping) and ("chunks" in row or "topk_chunks" in row)

def validate_retrieval_candidate_row(row: dict[str, Any]) -> None:
    required = [
        "sample_id",
        "query_id",
        "document_id",
        "query_text",
        "document_text",
        "strategy",
        "selector_version",
    ]
    missing = [key for key in required if not str(row.get(key, "") or "").strip()]
    if missing:
        raise ValueError(f"retrieval candidate row missing required keys: {', '.join(missing)}")

def validate_bm25_candidate_row(row: dict[str, Any]) -> None:
    required = [
        "sample_id",
        "query_id",
        "document_id",
        "query_text",
        "document_text",
        "strategy",
        "selector_version",
        "chunks",
    ]
    missing = [
        key
        for key in required
        if (
            (key == "chunks" and not has_chunk_field(row))
            or (
                key != "chunks"
                and (
                    key not in row
                    or row.get(key) is None
                    or (isinstance(row.get(key), str) and not str(row.get(key)).strip())
                )
            )
        )
    ]
    if missing:
        raise ValueError(f"bm25 candidate row missing required keys: {', '.join(missing)}")

def validate_embedding_candidate_row(row: dict[str, Any]) -> None:
    required = [
        "sample_id",
        "query_id",
        "document_id",
        "query_text",
        "document_text",
        "strategy",
        "selector_version",
        "chunks",
    ]
    missing = [
        key
        for key in required
        if (
            (key == "chunks" and not has_chunk_field(row))
            or (
                key != "chunks"
                and (
                    key not in row
                    or row.get(key) is None
                    or (isinstance(row.get(key), str) and not str(row.get(key)).strip())
                )
            )
        )
    ]
    if missing:
        raise ValueError(f"embedding candidate row missing required keys: {', '.join(missing)}")

def validate_rerank_result_row(row: dict[str, Any]) -> None:
    required = [
        "sample_id",
        "query_id",
        "document_id",
        "query_text",
        "document_text",
        "rerank_query_text",
        "rerank_scores",
        "chunks",
        "strategy",
        "rerank_cache_version",
    ]
    missing = [
        key
        for key in required
        if (
            (key == "chunks" and not has_chunk_field(row))
            or (
                key != "chunks"
                and (
                    key not in row
                    or row.get(key) is None
                    or (isinstance(row.get(key), str) and not str(row.get(key)).strip())
                )
            )
        )
    ]
    if missing:
        raise ValueError(f"rerank result row missing required keys: {', '.join(missing)}")

def retrieval_candidates_path(root: str | Path) -> Path:
    return Path(root) / "retrieval_candidates.jsonl"

def embedding_candidates_path(root: str | Path) -> Path:
    return Path(root) / "embedding_candidates.jsonl"

def bm25_candidates_path(root: str | Path) -> Path:
    return Path(root) / "bm25_candidates.jsonl"

def bm25_candidates_meta_path(root: str | Path) -> Path:
    return Path(root) / "bm25_candidates.meta.json"

def retrieval_candidates_meta_path(root: str | Path) -> Path:
    return Path(root) / "retrieval_candidates.meta.json"

def embedding_candidates_meta_path(root: str | Path) -> Path:
    return Path(root) / "embedding_candidates.meta.json"

def rerank_results_path(root: str | Path) -> Path:
    return Path(root) / "rerank_results.jsonl"

def rerank_results_meta_path(root: str | Path) -> Path:
    return Path(root) / "rerank_results.meta.json"

__all__ = [
    "bm25_candidates_meta_path",
    "bm25_candidates_path",
    "chunk_list",
    "embedding_candidates_meta_path",
    "embedding_candidates_path",
    "has_chunk_field",
    "rerank_results_meta_path",
    "rerank_results_path",
    "retrieval_candidates_meta_path",
    "retrieval_candidates_path",
    "validate_bm25_candidate_row",
    "validate_embedding_candidate_row",
    "validate_rerank_result_row",
    "validate_retrieval_candidate_row",
]

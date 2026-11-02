from __future__ import annotations
from pathlib import Path
from typing import Any
from script.retrieval.algorithms.embedding import embedded_bm25_row_from_embedding_candidate_row
from script.retrieval.algorithms.rrf import RRF_CANDIDATE_STRATEGY, build_rrf_candidate_row
from script.retrieval.settings import ChunkSelectionSettings, DEFAULT_CHUNK_SELECTOR_VERSION
from script.retrieval.utils.artifacts import (
    embedding_candidates_meta_path,
    embedding_candidates_path,
    retrieval_candidates_meta_path,
)
from script.retrieval.utils.numeric import safe_int
from script.io import count_jsonl_rows, iter_jsonl_objects, read_optional_json_object, remove_existing_files, write_json
from script.model_usage import build_cost_summary, elapsed_seconds_since, monotonic_seconds
from script.stream import StreamingRunWriter

def _reset_rrf_outputs(root: Path) -> None:
    remove_existing_files(
        root,
        (
            "retrieval_candidates.jsonl",
            "retrieval_candidates.meta.json",
            "events.jsonl",
            "progress.json",
            "summary.json",
        ),
    )

def _load_embedding_meta(root: str | Path) -> dict[str, Any]:
    return read_optional_json_object(embedding_candidates_meta_path(root))

def _source_model_usage(source_meta: dict[str, Any]) -> dict[str, Any]:
    model_usage = source_meta.get("model_usage", {})
    return dict(model_usage) if isinstance(model_usage, dict) else {}

def build_rrf_candidates(
    settings: ChunkSelectionSettings,
    *,
    embedding_root: str | Path,
) -> dict[str, object]:
    started_at = monotonic_seconds()
    embedding_file = embedding_candidates_path(embedding_root)
    if not embedding_file.exists():
        raise FileNotFoundError(f"embedding candidates file not found: {embedding_file}")
    root = Path(settings.artifacts_root)
    _reset_rrf_outputs(root)
    source_meta = _load_embedding_meta(embedding_root)
    writer = StreamingRunWriter(root)
    writer.write_event(
        "rrf_started",
        stage="rrf",
        strategy=RRF_CANDIDATE_STRATEGY,
        source_embedding_candidates_file=str(embedding_file),
    )
    total = count_jsonl_rows(embedding_file)
    completed_rows = 0
    retrieval_candidate_count = 0
    retrieval_topk_count = 0
    embedding_totals = {
        "embedding_model_call_count": 0,
        "embedding_model_request_count": 0,
        "embedding_model_input_count": 0,
        "embedding_prompt_tokens": 0,
        "embedding_system_prompt_tokens": 0,
        "embedding_user_prompt_tokens": 0,
    }
    selector_version = str(
        source_meta.get("selector_version", DEFAULT_CHUNK_SELECTOR_VERSION)
        or DEFAULT_CHUNK_SELECTOR_VERSION
    )
    for row in iter_jsonl_objects(embedding_file):
        output_row = build_rrf_candidate_row(
            embedded_bm25_row_from_embedding_candidate_row(row),
            settings=settings,
            selector_version=selector_version,
        )
        completed_rows += 1
        retrieval_candidate_count += safe_int(output_row.get("retrieval_candidate_count"), 0)
        retrieval_topk_count += safe_int(output_row.get("retrieval_topk_count"), 0)
        for key in embedding_totals:
            embedding_totals[key] += safe_int(output_row.get(key), 0)
        writer.write_stage_row("retrieval_candidates", output_row)
        writer.write_progress(
            stage="rrf",
            completed=completed_rows,
            total=total,
            strategy=RRF_CANDIDATE_STRATEGY,
        )
    totals = {
        key: safe_int(source_meta.get(key), value)
        for key, value in embedding_totals.items()
    }
    model_usage = _source_model_usage(source_meta)
    elapsed_seconds = elapsed_seconds_since(started_at)
    cost = build_cost_summary(elapsed_seconds=elapsed_seconds, model_usage=model_usage)
    meta = {
        "strategy": RRF_CANDIDATE_STRATEGY,
        "selector_version": selector_version,
        "source_embedding_candidates_file": str(embedding_file),
        "source_embedding_candidates_meta": source_meta,
        "source_bm25_candidates_file": str(source_meta.get("source_bm25_candidates_file", "") or ""),
        "source_bm25_candidates_meta": source_meta.get("source_bm25_candidates_meta", {}),
        "row_count": int(completed_rows),
        "window_size": safe_int(source_meta.get("window_size"), settings.window_size),
        "rrf_k": int(settings.rrf_k),
        "retrieval_candidate_count": int(retrieval_candidate_count),
        "retrieval_topk_count": int(retrieval_topk_count),
        **totals,
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    write_json(retrieval_candidates_meta_path(root), meta)
    writer.write_event(
        "rrf_finished",
        stage="rrf",
        strategy=RRF_CANDIDATE_STRATEGY,
        row_count=completed_rows,
    )
    summary = {
        "strategy": RRF_CANDIDATE_STRATEGY,
        "row_count": int(completed_rows),
        "artifacts_root": str(root),
        "source_embedding_candidates_file": str(embedding_file),
        "retrieval_candidate_count": int(retrieval_candidate_count),
        "retrieval_topk_count": int(retrieval_topk_count),
        **totals,
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    writer.write_summary(summary)
    return summary
__all__ = ["RRF_CANDIDATE_STRATEGY", "build_rrf_candidates"]

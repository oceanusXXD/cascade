from __future__ import annotations
from pathlib import Path
from typing import Any
from script.retrieval.utils.rows import build_selected_chunk_row, build_strategy_sample
from script.retrieval.algorithms.selection import cached_rerank_selection_from_row
from script.retrieval.utils.artifacts import (
    rerank_results_meta_path,
    rerank_results_path,
)
from script.retrieval.utils.numeric import safe_int
from script.retrieval.utils.usage import meta_int, meta_or_usage_int, source_model_usage
from script.retrieval.settings import DEFAULT_CHUNK_SELECTOR_VERSION
from script.chunk_artifacts import selected_chunks_meta_path
from script.io import count_jsonl_rows, iter_jsonl_objects, read_optional_json_object, remove_existing_files, write_json
from script.model_usage import build_cost_summary, elapsed_seconds_since, monotonic_seconds
from script.stream import StreamingRunWriter
CACHED_RERANK_SELECTION_STRATEGY = "rerank_cached"

def _reset_selectchunk_outputs(root: Path) -> None:
    remove_existing_files(
        root,
        (
            "selected_chunks.jsonl",
            "selected_chunks.meta.json",
            "events.jsonl",
            "progress.json",
            "summary.json",
        ),
    )

def select_chunks_from_rerank_cache(
    *,
    rerank_root: str | Path,
    artifacts_root: str | Path,
    selector_version: str = DEFAULT_CHUNK_SELECTOR_VERSION,
) -> dict[str, object]:
    started_at = monotonic_seconds()
    rerank_file = rerank_results_path(rerank_root)
    if not rerank_file.exists():
        raise FileNotFoundError(f"rerank results file not found: {rerank_file}")
    root = Path(artifacts_root)
    _reset_selectchunk_outputs(root)
    writer = StreamingRunWriter(root)
    writer.write_event(
        "selectchunk_started",
        stage="selectchunk",
        strategy=CACHED_RERANK_SELECTION_STRATEGY,
        source_rerank_results_file=str(rerank_file),
    )
    total_rows = count_jsonl_rows(rerank_file)
    selected_rows: list[dict[str, Any]] = []
    for row in iter_jsonl_objects(rerank_file):
        selection = cached_rerank_selection_from_row(row)
        selected_row = build_selected_chunk_row(
            build_strategy_sample(row),
            strategy_name=CACHED_RERANK_SELECTION_STRATEGY,
            selector_version=str(selector_version),
            selection=selection,
        )
        selected_rows.append(selected_row)
        writer.write_stage_row("selected_chunks", selected_row)
        writer.write_progress(
            stage="selectchunk",
            completed=len(selected_rows),
            total=total_rows,
            strategy=CACHED_RERANK_SELECTION_STRATEGY,
        )
    source_meta = read_optional_json_object(rerank_results_meta_path(rerank_root))
    model_usage = source_model_usage(source_meta)
    elapsed_seconds = elapsed_seconds_since(started_at)
    cost = build_cost_summary(elapsed_seconds=elapsed_seconds, model_usage=model_usage)
    meta = {
        "strategy": CACHED_RERANK_SELECTION_STRATEGY,
        "selector_version": str(selector_version),
        "row_count": int(len(selected_rows)),
        "source_rerank_results_file": str(rerank_file),
        "source_rerank_results_meta": source_meta,
        "embedding_model_call_count": meta_int(source_meta, "source_retrieval_candidates_meta", "embedding_model_call_count"),
        "embedding_model_request_count": meta_int(source_meta, "source_retrieval_candidates_meta", "embedding_model_request_count"),
        "embedding_model_input_count": meta_int(source_meta, "source_retrieval_candidates_meta", "embedding_model_input_count"),
        "embedding_prompt_tokens": meta_int(source_meta, "source_retrieval_candidates_meta", "embedding_prompt_tokens"),
        "embedding_system_prompt_tokens": meta_int(source_meta, "source_retrieval_candidates_meta", "embedding_system_prompt_tokens"),
        "embedding_user_prompt_tokens": meta_int(source_meta, "source_retrieval_candidates_meta", "embedding_user_prompt_tokens"),
        "rerank_model_call_count": safe_int(source_meta.get("rerank_model_call_count"), 0),
        "rerank_model_request_count": safe_int(source_meta.get("rerank_model_request_count"), 0),
        "rerank_model_input_count": safe_int(source_meta.get("rerank_model_input_count"), 0),
        "rerank_prompt_tokens": meta_or_usage_int(source_meta, "rerank_prompt_tokens"),
        "rerank_system_prompt_tokens": meta_or_usage_int(source_meta, "rerank_system_prompt_tokens"),
        "rerank_user_prompt_tokens": meta_or_usage_int(source_meta, "rerank_user_prompt_tokens"),
        "rerank_completion_tokens": meta_or_usage_int(source_meta, "rerank_completion_tokens"),
        "full_rerank_candidate_count": int(
            sum(int(row.get("full_rerank_candidate_count", 0) or 0) for row in selected_rows)
        ),
        "rerank_saved_candidate_count": int(
            sum(int(row.get("rerank_saved_candidate_count", 0) or 0) for row in selected_rows)
        ),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    write_json(selected_chunks_meta_path(root), meta)
    writer.write_event(
        "selectchunk_finished",
        stage="selectchunk",
        strategy=CACHED_RERANK_SELECTION_STRATEGY,
        row_count=len(selected_rows),
    )
    summary = {
        "strategy": CACHED_RERANK_SELECTION_STRATEGY,
        "selector_version": str(selector_version),
        "row_count": int(len(selected_rows)),
        "artifacts_root": str(root),
        "source_rerank_results_file": str(rerank_file),
        "embedding_model_input_count": int(meta["embedding_model_input_count"]),
        "embedding_prompt_tokens": int(meta["embedding_prompt_tokens"]),
        "embedding_system_prompt_tokens": int(meta["embedding_system_prompt_tokens"]),
        "embedding_user_prompt_tokens": int(meta["embedding_user_prompt_tokens"]),
        "rerank_model_input_count": int(meta["rerank_model_input_count"]),
        "rerank_prompt_tokens": int(meta["rerank_prompt_tokens"]),
        "rerank_system_prompt_tokens": int(meta["rerank_system_prompt_tokens"]),
        "rerank_user_prompt_tokens": int(meta["rerank_user_prompt_tokens"]),
        "rerank_completion_tokens": int(meta["rerank_completion_tokens"]),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    writer.write_summary(summary)
    return summary
__all__ = [
    "CACHED_RERANK_SELECTION_STRATEGY",
    "select_chunks_from_rerank_cache",
]

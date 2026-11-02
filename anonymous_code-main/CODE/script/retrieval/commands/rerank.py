from __future__ import annotations
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from script.retrieval.algorithms.rerank import (
    RERANK_CACHE_STRATEGY,
    RERANK_CACHE_VERSION,
    build_rerank_result_row,
    build_rerank_row_job,
)
from script.retrieval.algorithms.rerank import score_query_document_pairs_with_usage
from script.retrieval.utils.artifacts import (
    rerank_results_meta_path,
    retrieval_candidates_path,
)
from script.retrieval.utils.usage import build_rerank_model_usage
from script.retrieval.settings import ChunkSelectionSettings
from script.io import count_jsonl_rows, iter_jsonl_objects, read_optional_json_object, remove_existing_files, write_json
from script.model_usage import build_cost_summary, elapsed_seconds_since, monotonic_seconds
from script.stream import StreamingRunWriter
from script.vllm import vllm_score_query_document_pairs

def _reset_rerank_outputs(root: Path) -> None:
    remove_existing_files(
        root,
        (
            "rerank_results.jsonl",
            "rerank_results.meta.json",
            "selected_chunks.jsonl",
            "selected_chunks.meta.json",
            "events.jsonl",
            "progress.json",
            "summary.json",
        ),
    )

def _load_source_meta(retrieval_file: Path) -> dict[str, Any]:
    return read_optional_json_object(retrieval_file.with_name("retrieval_candidates.meta.json"))

def rerank_retrieval_candidates(
    settings: ChunkSelectionSettings,
    *,
    retrieval_root: str | Path,
    rerank_pair_score_func: Callable[..., Sequence[float]] = vllm_score_query_document_pairs,
) -> dict[str, object]:
    started_at = monotonic_seconds()
    if not str(settings.reranker_model or "").strip():
        raise ValueError("reranker_model is required for rerank")
    if not str(settings.reranker_base_url or "").strip():
        raise ValueError("reranker_base_url is required for rerank")
    retrieval_file = retrieval_candidates_path(retrieval_root)
    if not retrieval_file.exists():
        raise FileNotFoundError(f"retrieval candidates file not found: {retrieval_file}")
    artifacts_root = Path(settings.artifacts_root)
    _reset_rerank_outputs(artifacts_root)
    writer = StreamingRunWriter(artifacts_root)
    writer.write_event(
        "rerank_started",
        stage="rerank",
        strategy=RERANK_CACHE_STRATEGY,
        source_retrieval_candidates_file=str(retrieval_file),
    )
    total_rows = count_jsonl_rows(retrieval_file)
    request_batch_size = max(1, int(settings.rerank_request_batch_size))
    completed_rows = 0
    rerank_model_call_count = 0
    rerank_model_request_count = 0
    rerank_model_input_count = 0
    rerank_prompt_tokens = 0
    rerank_system_prompt_tokens = 0
    rerank_user_prompt_tokens = 0
    rerank_completion_tokens = 0
    pending_rows: list[dict[str, Any]] = []
    pending_pairs: list[tuple[str, str]] = []
    def flush_pending_rows() -> None:
        nonlocal completed_rows
        nonlocal rerank_model_call_count, rerank_model_request_count, rerank_model_input_count
        nonlocal rerank_prompt_tokens, rerank_system_prompt_tokens, rerank_user_prompt_tokens
        nonlocal rerank_completion_tokens
        if not pending_rows:
            return
        score_result = score_query_document_pairs_with_usage(
            pairs=pending_pairs,
            settings=settings,
            rerank_pair_score_func=rerank_pair_score_func,
        )
        for index, state in enumerate(pending_rows):
            pair_start = int(state["pair_start"])
            pair_count = int(state["pair_count"])
            row_scores = score_result.scores[pair_start : pair_start + pair_count]
            result_row = build_rerank_result_row(
                state["row"],
                rerank_query_text=str(state["rerank_query_text"]),
                rerank_query_source=str(state["rerank_query_source"]),
                scores=row_scores,
                settings=settings,
                scoring_result=score_result,
                assign_model_call_usage=index == 0,
            )
            completed_rows += 1
            rerank_model_call_count += int(result_row.get("rerank_model_call_count", 0) or 0)
            rerank_model_request_count += int(result_row.get("rerank_model_request_count", 0) or 0)
            rerank_model_input_count += int(result_row.get("rerank_model_input_count", 0) or 0)
            rerank_prompt_tokens += int(result_row.get("rerank_prompt_tokens", 0) or 0)
            rerank_system_prompt_tokens += int(result_row.get("rerank_system_prompt_tokens", 0) or 0)
            rerank_user_prompt_tokens += int(result_row.get("rerank_user_prompt_tokens", 0) or 0)
            rerank_completion_tokens += int(result_row.get("rerank_completion_tokens", 0) or 0)
            writer.write_stage_row("rerank_results", result_row)
            writer.write_progress(
                stage="rerank",
                completed=completed_rows,
                total=total_rows,
                strategy=RERANK_CACHE_STRATEGY,
            )
        pending_rows.clear()
        pending_pairs.clear()
    for row in iter_jsonl_objects(retrieval_file):
        job = build_rerank_row_job(row)
        row_pairs = job.pairs
        if pending_pairs and len(pending_pairs) + len(row_pairs) > request_batch_size:
            flush_pending_rows()
        pair_start = len(pending_pairs)
        pending_pairs.extend(row_pairs)
        pending_rows.append(
            {
                "row": job.row,
                "rerank_query_text": job.rerank_query_text,
                "rerank_query_source": job.rerank_query_source,
                "pair_start": pair_start,
                "pair_count": len(row_pairs),
            }
        )
        if pending_pairs and len(pending_pairs) >= request_batch_size:
            flush_pending_rows()
    flush_pending_rows()
    model_usage = build_rerank_model_usage(
        settings,
        call_count=rerank_model_call_count,
        request_count=rerank_model_request_count,
        input_count=rerank_model_input_count,
        prompt_tokens=rerank_prompt_tokens,
        system_prompt_tokens=rerank_system_prompt_tokens,
        user_prompt_tokens=rerank_user_prompt_tokens,
        completion_tokens=rerank_completion_tokens,
    )
    source_meta = _load_source_meta(retrieval_file)
    elapsed_seconds = elapsed_seconds_since(started_at)
    cost = build_cost_summary(elapsed_seconds=elapsed_seconds, model_usage=model_usage)
    meta = {
        "strategy": RERANK_CACHE_STRATEGY,
        "rerank_cache_version": RERANK_CACHE_VERSION,
        "source_retrieval_candidates_file": str(retrieval_file),
        "source_retrieval_candidates_meta": source_meta,
        "row_count": int(completed_rows),
        "rerank_model_call_count": rerank_model_call_count,
        "rerank_model_request_count": rerank_model_request_count,
        "rerank_model_input_count": rerank_model_input_count,
        "rerank_prompt_tokens": rerank_prompt_tokens,
        "rerank_system_prompt_tokens": rerank_system_prompt_tokens,
        "rerank_user_prompt_tokens": rerank_user_prompt_tokens,
        "rerank_completion_tokens": rerank_completion_tokens,
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    write_json(rerank_results_meta_path(artifacts_root), meta)
    writer.write_event("rerank_finished", stage="rerank", strategy=RERANK_CACHE_STRATEGY, row_count=completed_rows)
    summary = {
        "strategy": RERANK_CACHE_STRATEGY,
        "rerank_cache_version": RERANK_CACHE_VERSION,
        "row_count": int(completed_rows),
        "artifacts_root": str(artifacts_root),
        "source_retrieval_candidates_file": str(retrieval_file),
        "rerank_model_call_count": rerank_model_call_count,
        "rerank_model_request_count": rerank_model_request_count,
        "rerank_model_input_count": rerank_model_input_count,
        "rerank_prompt_tokens": rerank_prompt_tokens,
        "rerank_system_prompt_tokens": rerank_system_prompt_tokens,
        "rerank_user_prompt_tokens": rerank_user_prompt_tokens,
        "rerank_completion_tokens": rerank_completion_tokens,
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    writer.write_summary(summary)
    return summary
__all__ = [
    "RERANK_CACHE_STRATEGY",
    "RERANK_CACHE_VERSION",
    "rerank_retrieval_candidates",
]

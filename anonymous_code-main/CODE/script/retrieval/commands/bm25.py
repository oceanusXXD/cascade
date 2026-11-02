from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from script.retrieval.algorithms.bm25 import BM25_CANDIDATE_STRATEGY, build_bm25_candidate_row
from script.retrieval.utils.artifacts import bm25_candidates_meta_path, validate_bm25_candidate_row
from script.retrieval.utils.cache import QueryGenerationCacheLookup, load_query_generation_cache
from script.retrieval.utils.rows import build_strategy_sample
from script.retrieval.utils.text import resolve_row_effective_query_text
from script.retrieval.settings import ChunkSelectionSettings, DEFAULT_CHUNK_SELECTOR_VERSION
from script.data import build_sample_rows
from script.io import remove_existing_files, write_json
from script.model_usage import build_cost_summary, elapsed_seconds_since, monotonic_seconds
from script.stream import StreamingRunWriter
from script.text import SENTENCE_TOKENIZER_NAME

def _selection_batch_size() -> int:
    try:
        return max(1, int(str(os.environ.get("CASCADE_BATCHPROCESS_SELECT_BATCH_SIZE", "512")).strip()))
    except (TypeError, ValueError):
        return 512

def _load_enabled_query_generation_cache(
    *,
    settings: ChunkSelectionSettings,
    kind: str,
) -> QueryGenerationCacheLookup | None:
    if kind == "query_rewrite" and not settings.use_query_rewrite_cache:
        return None
    if kind == "hyde" and not settings.use_hyde_cache:
        return None
    return load_query_generation_cache(kind, queries_path=settings.queries)

def _reset_bm25_outputs(root: Path) -> None:
    remove_existing_files(
        root,
        (
            "bm25_candidates.jsonl",
            "bm25_candidates.meta.json",
            "events.jsonl",
            "progress.json",
            "summary.json",
        ),
    )

def build_bm25_candidates(settings: ChunkSelectionSettings) -> dict[str, object]:
    started_at = monotonic_seconds()
    root = Path(settings.artifacts_root)
    _reset_bm25_outputs(root)
    writer = StreamingRunWriter(root)
    writer.write_event("bm25_started", stage="bm25", strategy=BM25_CANDIDATE_STRATEGY)
    rewrite_cache = _load_enabled_query_generation_cache(settings=settings, kind="query_rewrite")
    hyde_cache = _load_enabled_query_generation_cache(settings=settings, kind="hyde")
    input_rows = build_sample_rows(
        documents_path=settings.documents,
        queries_path=settings.queries,
        ground_truth_path=settings.ground_truth,
    )
    total = len(input_rows)
    rows: list[dict[str, Any]] = []
    batch_size = _selection_batch_size()
    pending: list[dict[str, Any]] = []
    def flush() -> None:
        if not pending:
            return
        for row in pending:
            effective_query = resolve_row_effective_query_text(row)
            sample = build_strategy_sample(
                row,
                query_rewrites=rewrite_cache.get(effective_query) if rewrite_cache is not None else (),
                hyde_hypotheses=hyde_cache.get(effective_query) if hyde_cache is not None else (),
            )
            bm25_row = build_bm25_candidate_row(
                sample,
                window_size=int(settings.window_size),
                rrf_k=int(settings.rrf_k),
                selector_version=DEFAULT_CHUNK_SELECTOR_VERSION,
            )
            validate_bm25_candidate_row(bm25_row)
            rows.append(bm25_row)
            writer.write_stage_row("bm25_candidates", bm25_row)
            writer.write_progress(stage="bm25", completed=len(rows), total=total, strategy=BM25_CANDIDATE_STRATEGY)
        pending.clear()
    for row in input_rows:
        pending.append(row)
        if len(pending) >= batch_size:
            flush()
    flush()
    candidate_count = int(sum(int(row.get("retrieval_candidate_count", 0) or 0) for row in rows))
    elapsed_seconds = elapsed_seconds_since(started_at)
    cost = build_cost_summary(elapsed_seconds=elapsed_seconds)
    meta = {
        "strategy": BM25_CANDIDATE_STRATEGY,
        "selector_version": DEFAULT_CHUNK_SELECTOR_VERSION,
        "sentence_tokenizer": SENTENCE_TOKENIZER_NAME,
        "row_count": int(len(rows)),
        "window_size": int(settings.window_size),
        "rrf_k": int(settings.rrf_k),
        "bm25_candidate_count": int(candidate_count),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
    }
    write_json(bm25_candidates_meta_path(root), meta)
    writer.write_event("bm25_finished", stage="bm25", strategy=BM25_CANDIDATE_STRATEGY, row_count=len(rows))
    summary = {
        **meta,
        "artifacts_root": str(root),
    }
    writer.write_summary(summary)
    return summary
__all__ = ["BM25_CANDIDATE_STRATEGY", "build_bm25_candidates"]

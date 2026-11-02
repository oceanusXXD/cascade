from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from algorithm.settings import CascadeSettings, DEFAULT_STEP1_BATCH_PROTOCOL_VERSION
from algorithm.common.artifacts import validate_evidence_row, validate_step1_batch_row
from algorithm.common.embeddings import PAIR_EMBEDDING_VERSION, write_evidence_row_embeddings_npy
from script.chunk_artifacts import load_selected_chunks, selected_chunks_meta_path
from script.data import build_sample_rows
from script.io import read_json, remove_existing_files, write_json
from script.model_usage import (
    build_cost_summary,
    elapsed_seconds_since,
    merge_model_usage_entries,
    model_usage_entries_from_payload,
    model_usage_entry,
    monotonic_seconds,
)
from script.stream import StreamingRunWriter
from script.task_schema import (
    resolve_default_task_text,
    resolve_embedding_payload,
    resolve_prompt_payload,
)

def _reset_document_outputs(root: Path) -> None:
    remove_existing_files(
        root,
        (
            "evidence_rows.jsonl",
            "stage1_pair_embeddings.npy",
            "stage1_pair_embeddings.meta.json",
            "stage2_pair_embeddings.npy",
            "stage2_pair_embeddings.meta.json",
            "evidence_pair_embeddings.npy",
            "evidence_pair_embeddings.meta.json",
            "events.jsonl",
            "progress.json",
            "summary.json",
        ),
    )

def prepare_document_artifacts(
    settings: CascadeSettings,
    *,
    writer: StreamingRunWriter,
    embed_texts_func,
) -> dict[str, object]:
    started_at = monotonic_seconds()
    _reset_document_outputs(writer.root)
    writer.write_event("prepare_started", stage="prepare", mode="document")
    sample_rows = build_sample_rows(
        documents_path=settings.documents,
        queries_path=settings.queries,
        ground_truth_path=settings.ground_truth or None,
    )
    evidence_rows: list[dict[str, object]] = []
    total = len(sample_rows)
    for index, row in enumerate(sample_rows, start=1):
        default_task_text = resolve_default_task_text(row, fallback=str(row["query_text"]))
        prompt_payload = resolve_prompt_payload(
            row,
            default_task_text=default_task_text,
            default_evidence_text=str(row["document_text"]),
        )
        embedding_payload = resolve_embedding_payload(
            row,
            default_task_text=default_task_text,
            default_evidence_text=str(row["document_text"]),
        )
        evidence_row = {
            "sample_id": str(row["sample_id"]),
            "query_id": str(row["query_id"]),
            "document_id": str(row["document_id"]),
            "query_text": str(row["query_text"]),
            "raw_query_text": str(row.get("raw_query_text", row["query_text"])),
            "document_text": str(row["document_text"]),
            "evidence_mode": "document",
            "evidence_text": str(row["document_text"]),
            "label": str(row.get("label", "")),
            "stage2_prediction": str(row.get("stage2_prediction", "")),
            "stage2_routing_score": row.get("stage2_routing_score", ""),
            "claim": str(row.get("claim_text", "")),
            "claim_text": str(row.get("claim_text", "")),
            "prompt_style": str(row.get("prompt_style", "query_document")),
            "prompt_task_label": prompt_payload["task_label"],
            "prompt_evidence_label": prompt_payload["evidence_label"],
            "prompt_task_text": prompt_payload["task_text"],
            "prompt_instruction_text": prompt_payload["instruction_text"],
            "prompt_evidence_text": prompt_payload["evidence_text"],
            "embedding_task_text": embedding_payload["task_text"],
            "embedding_instruction_text": embedding_payload["instruction_text"],
            "embedding_evidence_text": embedding_payload["evidence_text"],
        }
        validate_evidence_row(evidence_row)
        evidence_rows.append(evidence_row)
        writer.write_stage_row("evidence_rows", evidence_row)
        writer.write_progress(stage="prepare", completed=index, total=total, mode="document")
    pair_embedding_usage: dict[str, int] = {}
    root = writer.root
    def write_pair_embedding_progress(progress: dict[str, int]) -> None:
        writer.write_event(
            "pair_embedding_progress",
            stage="stage1_pair_embedding",
            embedding_stage="stage1",
            mode="document",
            completed=int(progress.get("completed", 0) or 0),
            total=int(progress.get("total", 0) or 0),
            request_count=int(progress.get("request_count", 0) or 0),
            input_count=int(progress.get("input_count", 0) or 0),
        )
        writer.write_progress(
            stage="stage1_pair_embedding",
            completed=int(progress.get("completed", 0) or 0),
            total=int(progress.get("total", 0) or 0),
            mode="document",
            embedding_stage="stage1",
            request_count=int(progress.get("request_count", 0) or 0),
            input_count=int(progress.get("input_count", 0) or 0),
        )
    pair_embedding_row_count, pair_embedding_dim = write_evidence_row_embeddings_npy(
        "document",
        evidence_rows,
        root / "stage1_pair_embeddings.npy",
        stage="stage1",
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        embed_texts_func=embed_texts_func,
        usage_callback=pair_embedding_usage.update,
        progress_callback=write_pair_embedding_progress,
    )
    copyfile(root / "stage1_pair_embeddings.npy", root / "stage2_pair_embeddings.npy")
    copyfile(root / "stage1_pair_embeddings.npy", root / "evidence_pair_embeddings.npy")
    model_usage = merge_model_usage_entries(
        [
            model_usage_entry(
                scope="prepare.stage1_pair_embedding",
                stage="prepare",
                model=str(settings.embedding_model),
                base_url=str(settings.embedding_base_url),
                call_count=int(pair_embedding_usage.get("call_count", 0) or 0),
                request_count=int(pair_embedding_usage.get("request_count", 0) or 0),
                input_count=int(pair_embedding_usage.get("input_count", 0) or 0),
                prompt_tokens=int(pair_embedding_usage.get("prompt_tokens", 0) or 0),
                system_prompt_tokens=int(pair_embedding_usage.get("system_prompt_tokens", 0) or 0),
                user_prompt_tokens=int(pair_embedding_usage.get("user_prompt_tokens", 0) or 0),
                completion_tokens=0,
                token_accounting_mode="estimated_text",
            )
        ]
    )
    elapsed_seconds = elapsed_seconds_since(started_at)
    cost = build_cost_summary(elapsed_seconds=elapsed_seconds, model_usage=model_usage)
    meta = {
        "mode": "document",
        "sample_count": int(len(evidence_rows)),
        "row_count": int(pair_embedding_row_count),
        "dimension": int(pair_embedding_dim),
        "embedding_model": str(settings.embedding_model),
        "pair_embedding_version": str(PAIR_EMBEDDING_VERSION),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    stage1_meta = {**meta, "embedding_stage": "stage1", "embedding_source": "stage1_input"}
    stage2_meta = {
        **meta,
        "embedding_stage": "stage2",
        "embedding_source": "stage2_input",
        "reused_from_stage": "stage1",
    }
    write_json(root / "stage1_pair_embeddings.meta.json", stage1_meta)
    write_json(root / "stage2_pair_embeddings.meta.json", stage2_meta)
    write_json(
        root / "evidence_pair_embeddings.meta.json",
        {**stage1_meta, "legacy_alias_for": "stage1_pair_embeddings"},
    )
    writer.write_event("prepare_finished", stage="prepare", mode="document", row_count=len(evidence_rows))
    summary = {
        "mode": "document",
        "row_count": int(len(evidence_rows)),
        "stage1_pair_embedding_dim": int(pair_embedding_dim),
        "stage2_pair_embedding_dim": int(pair_embedding_dim),
        "pair_embedding_dim": int(pair_embedding_dim),
        "artifacts_root": str(root),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    writer.write_summary(summary)
    return summary
_CHUNK_STRATEGY = "selected_chunks"

def _reset_chunk_outputs(root: Path) -> None:
    remove_existing_files(
        root,
        (
            "evidence_rows.jsonl",
            "step1_batch_inputs.jsonl",
            "stage1_pair_embeddings.npy",
            "stage1_pair_embeddings.meta.json",
            "stage2_pair_embeddings.npy",
            "stage2_pair_embeddings.meta.json",
            "evidence_pair_embeddings.npy",
            "evidence_pair_embeddings.meta.json",
        ),
    )

def _build_step1_batch_item(row: dict[str, object], display_index: int) -> dict[str, object]:
    payload = resolve_prompt_payload(
        row,
        default_task_text=resolve_default_task_text(row),
        default_evidence_text=str(row.get("selected_chunk_text", "") or ""),
    )
    return {
        "sample_id": str(row["sample_id"]),
        "claim": str(row.get("claim", "") or ""),
        "task_text": payload["task_text"],
        "selected_chunk_text": str(row["selected_chunk_text"]),
        "evidence_text": payload["evidence_text"],
        "display_index": int(display_index),
    }

def _build_step1_batch_inputs(settings: CascadeSettings, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    batch_rows: list[dict[str, object]] = []
    for query_id in sorted(grouped):
        query_rows = list(grouped[query_id])
        for batch_index, start in enumerate(
            range(0, len(query_rows), int(settings.step1_batch_size)),
            start=1,
        ):
            batch = query_rows[start : start + int(settings.step1_batch_size)]
            batch_row = {
                "batch_id": f"{query_id}:{batch_index:04d}",
                "query_id": str(query_id),
                "query_text": str(batch[0]["query_text"]) if batch else "",
                "raw_query_text": str(batch[0].get("raw_query_text", batch[0]["query_text"])) if batch else "",
                "prompt_task_label": str(batch[0].get("prompt_task_label", "Query")) if batch else "Query",
                "prompt_evidence_label": str(batch[0].get("prompt_evidence_label", "Document")) if batch else "Document",
                "prompt_instruction_text": str(batch[0].get("prompt_instruction_text", "")) if batch else "",
                "sample_ids": [str(row["sample_id"]) for row in batch],
                "prompt_template_version": DEFAULT_STEP1_BATCH_PROTOCOL_VERSION,
                "items": [
                    _build_step1_batch_item(row, display_index)
                    for display_index, row in enumerate(batch, start=1)
                ],
            }
            validate_step1_batch_row(batch_row)
            batch_rows.append(batch_row)
    return batch_rows

def _build_evidence_rows_from_selected_chunks(
    selected_rows: list[dict],
    *,
    writer: StreamingRunWriter,
) -> list[dict[str, object]]:
    evidence_rows: list[dict[str, object]] = []
    total = len(selected_rows)
    for index, raw in enumerate(selected_rows, start=1):
        selected_chunk_text = str(raw["selected_chunk_text"])
        default_task_text = resolve_default_task_text(raw)
        payload_source = {
            **raw,
            "prompt_evidence_text": selected_chunk_text,
            "embedding_evidence_text": selected_chunk_text,
        }
        prompt_payload = resolve_prompt_payload(
            payload_source,
            default_task_text=default_task_text,
            default_evidence_text=selected_chunk_text,
        )
        embedding_payload = resolve_embedding_payload(
            payload_source,
            default_task_text=default_task_text,
            default_evidence_text=selected_chunk_text,
        )
        evidence_row = {
            "sample_id": str(raw["sample_id"]),
            "query_id": str(raw["query_id"]),
            "document_id": str(raw["document_id"]),
            "query_text": str(raw["query_text"]),
            "raw_query_text": str(raw.get("raw_query_text", raw["query_text"])),
            "document_text": str(raw["document_text"]),
            "evidence_mode": "chunk",
            "evidence_text": selected_chunk_text,
            "selected_chunk_text": selected_chunk_text,
            "selected_chunk_indices": list(raw.get("selected_chunk_indices", []) or []),
            "label": str(raw.get("label", "")),
            "stage2_prediction": str(raw.get("stage2_prediction", "")),
            "stage2_routing_score": raw.get("stage2_routing_score", ""),
            "claim": str(raw.get("claim", raw["query_text"])),
            "claim_text": str(raw.get("claim_text", raw.get("claim", raw["query_text"]))),
            "prompt_style": str(raw.get("prompt_style", "query_document")),
            "prompt_task_label": prompt_payload["task_label"],
            "prompt_evidence_label": prompt_payload["evidence_label"],
            "prompt_task_text": prompt_payload["task_text"],
            "prompt_instruction_text": prompt_payload["instruction_text"],
            "prompt_evidence_text": prompt_payload["evidence_text"],
            "embedding_task_text": embedding_payload["task_text"],
            "embedding_instruction_text": embedding_payload["instruction_text"],
            "embedding_evidence_text": embedding_payload["evidence_text"],
            "strategy": str(raw.get("strategy", _CHUNK_STRATEGY)),
        }
        validate_evidence_row(evidence_row)
        evidence_rows.append(evidence_row)
        writer.write_stage_row("evidence_rows", evidence_row)
        writer.write_progress(stage="prepare", completed=index, total=total, mode="chunk", strategy=_CHUNK_STRATEGY)
    return evidence_rows

def _write_chunk_prepared_artifacts(
    settings: CascadeSettings,
    *,
    writer: StreamingRunWriter,
    selected_rows: list[dict],
    selected_meta: dict[str, object],
    embed_texts_func,
    source_selected_chunks_file: str = "",
) -> dict[str, object]:
    started_at = monotonic_seconds()
    _reset_chunk_outputs(writer.root)
    writer.write_event(
        "prepare_started",
        stage="prepare",
        mode="chunk",
        strategy=str(selected_meta.get("strategy", _CHUNK_STRATEGY)),
        source="selected_chunks",
    )
    evidence_rows = _build_evidence_rows_from_selected_chunks(selected_rows, writer=writer)
    batch_rows = _build_step1_batch_inputs(settings, evidence_rows)
    for batch_row in batch_rows:
        writer.write_stage_row("step1_batch_inputs", batch_row)
    stage1_pair_embedding_usage: dict[str, int] = {}
    stage2_pair_embedding_usage: dict[str, int] = {}
    root = writer.root

    def make_pair_embedding_progress_writer(embedding_stage: str):
        def write_pair_embedding_progress(progress: dict[str, int]) -> None:
            writer.write_event(
                "pair_embedding_progress",
                stage=f"{embedding_stage}_pair_embedding",
                embedding_stage=embedding_stage,
                mode="chunk",
                completed=int(progress.get("completed", 0) or 0),
                total=int(progress.get("total", 0) or 0),
                request_count=int(progress.get("request_count", 0) or 0),
                input_count=int(progress.get("input_count", 0) or 0),
            )
            writer.write_progress(
                stage=f"{embedding_stage}_pair_embedding",
                completed=int(progress.get("completed", 0) or 0),
                total=int(progress.get("total", 0) or 0),
                mode="chunk",
                embedding_stage=embedding_stage,
                strategy=str(selected_meta.get("strategy", _CHUNK_STRATEGY)),
                request_count=int(progress.get("request_count", 0) or 0),
                input_count=int(progress.get("input_count", 0) or 0),
            )

        return write_pair_embedding_progress

    stage1_pair_embedding_row_count, stage1_pair_embedding_dim = write_evidence_row_embeddings_npy(
        "chunk",
        evidence_rows,
        root / "stage1_pair_embeddings.npy",
        stage="stage1",
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        embed_texts_func=embed_texts_func,
        usage_callback=stage1_pair_embedding_usage.update,
        progress_callback=make_pair_embedding_progress_writer("stage1"),
    )
    stage2_pair_embedding_row_count, stage2_pair_embedding_dim = write_evidence_row_embeddings_npy(
        "chunk",
        evidence_rows,
        root / "stage2_pair_embeddings.npy",
        stage="stage2",
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        embed_texts_func=embed_texts_func,
        usage_callback=stage2_pair_embedding_usage.update,
        progress_callback=make_pair_embedding_progress_writer("stage2"),
    )
    copyfile(root / "stage1_pair_embeddings.npy", root / "evidence_pair_embeddings.npy")
    model_usage = merge_model_usage_entries(
        [
            *model_usage_entries_from_payload(selected_meta),
            model_usage_entry(
                scope="prepare.stage1_pair_embedding",
                stage="prepare",
                model=str(settings.embedding_model),
                base_url=str(settings.embedding_base_url),
                call_count=int(stage1_pair_embedding_usage.get("call_count", 0) or 0),
                request_count=int(stage1_pair_embedding_usage.get("request_count", 0) or 0),
                input_count=int(stage1_pair_embedding_usage.get("input_count", 0) or 0),
                prompt_tokens=int(stage1_pair_embedding_usage.get("prompt_tokens", 0) or 0),
                system_prompt_tokens=int(stage1_pair_embedding_usage.get("system_prompt_tokens", 0) or 0),
                user_prompt_tokens=int(stage1_pair_embedding_usage.get("user_prompt_tokens", 0) or 0),
                completion_tokens=0,
                token_accounting_mode="estimated_text",
            ),
            model_usage_entry(
                scope="prepare.stage2_pair_embedding",
                stage="prepare",
                model=str(settings.embedding_model),
                base_url=str(settings.embedding_base_url),
                call_count=int(stage2_pair_embedding_usage.get("call_count", 0) or 0),
                request_count=int(stage2_pair_embedding_usage.get("request_count", 0) or 0),
                input_count=int(stage2_pair_embedding_usage.get("input_count", 0) or 0),
                prompt_tokens=int(stage2_pair_embedding_usage.get("prompt_tokens", 0) or 0),
                system_prompt_tokens=int(stage2_pair_embedding_usage.get("system_prompt_tokens", 0) or 0),
                user_prompt_tokens=int(stage2_pair_embedding_usage.get("user_prompt_tokens", 0) or 0),
                completion_tokens=0,
                token_accounting_mode="estimated_text",
            ),
        ]
    )
    elapsed_seconds = elapsed_seconds_since(started_at)
    cost = build_cost_summary(elapsed_seconds=elapsed_seconds, model_usage=model_usage)
    meta = {
        "mode": "chunk",
        "sample_count": int(len(evidence_rows)),
        "row_count": int(stage1_pair_embedding_row_count),
        "dimension": int(stage1_pair_embedding_dim),
        "step1_batch_size": int(settings.step1_batch_size),
        "prompt_template_version": DEFAULT_STEP1_BATCH_PROTOCOL_VERSION,
        "selector_version": str(selected_meta.get("selector_version", "") or ""),
        "strategy": str(selected_meta.get("strategy", _CHUNK_STRATEGY)),
        "embedding_model": str(settings.embedding_model),
        "pair_embedding_version": str(PAIR_EMBEDDING_VERSION),
        "source_selected_chunks_file": str(source_selected_chunks_file),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    stage1_meta = {**meta, "embedding_stage": "stage1", "embedding_source": "stage1_input"}
    stage2_meta = {
        **meta,
        "embedding_stage": "stage2",
        "embedding_source": "stage2_input",
        "row_count": int(stage2_pair_embedding_row_count),
        "dimension": int(stage2_pair_embedding_dim),
    }
    write_json(root / "stage1_pair_embeddings.meta.json", stage1_meta)
    write_json(root / "stage2_pair_embeddings.meta.json", stage2_meta)
    write_json(
        root / "evidence_pair_embeddings.meta.json",
        {**stage1_meta, "legacy_alias_for": "stage1_pair_embeddings"},
    )
    writer.write_event("prepare_finished", stage="prepare", mode="chunk", row_count=len(evidence_rows))
    summary = {
        "mode": "chunk",
        "strategy": str(selected_meta.get("strategy", _CHUNK_STRATEGY)),
        "row_count": int(len(evidence_rows)),
        "stage1_pair_embedding_dim": int(stage1_pair_embedding_dim),
        "stage2_pair_embedding_dim": int(stage2_pair_embedding_dim),
        "pair_embedding_dim": int(stage1_pair_embedding_dim),
        "artifacts_root": str(root),
        "source_selected_chunks_file": str(source_selected_chunks_file),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "model_usage": model_usage,
    }
    writer.write_summary(summary)
    return summary

def prepare_chunk_artifacts(
    settings: CascadeSettings,
    *,
    writer: StreamingRunWriter,
    embed_texts_func,
) -> dict[str, object]:
    selected_rows = load_selected_chunks(settings.artifacts_root)
    selected_meta = read_json(selected_chunks_meta_path(settings.artifacts_root))
    return _write_chunk_prepared_artifacts(
        settings,
        writer=writer,
        selected_rows=selected_rows,
        selected_meta=selected_meta,
        embed_texts_func=embed_texts_func,
        source_selected_chunks_file=str(Path(settings.artifacts_root) / "selected_chunks.jsonl"),
    )
__all__ = [
    "prepare_chunk_artifacts",
    "prepare_document_artifacts",
]

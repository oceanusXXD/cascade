from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from algorithm.common.embeddings import PAIR_EMBEDDING_VERSION
from algorithm.common.runtime import Action, Sample, Source
from algorithm.settings import CascadeSettings
from script.data import load_ground_truth_index
from script.io import read_json, read_jsonl, read_jsonl_lenient, write_json, write_jsonl
from script.labels import normalize_binary_label
from script.task_schema import resolve_default_task_text, resolve_embedding_payload, resolve_prompt_payload

class ArtifactContractError(RuntimeError):
    pass

def require_mapping_keys(row: dict[str, Any], required_keys: list[str], *, row_name: str) -> None:
    missing = [key for key in required_keys if not str(row.get(key, "") or "").strip()]
    if missing:
        raise ValueError(f"{row_name} missing required keys: {', '.join(missing)}")

def validate_evidence_row(row: dict[str, Any]) -> None:
    require_mapping_keys(
        row,
        ["sample_id", "query_id", "document_id", "query_text", "document_text", "evidence_mode", "evidence_text"],
        row_name="evidence row",
    )

def validate_step1_batch_row(row: dict[str, Any]) -> None:
    require_mapping_keys(
        row,
        ["batch_id", "query_id", "query_text"],
        row_name="step1 batch row",
    )
    items = list(row.get("items", []) or [])
    if not items:
        raise ValueError("step1 batch row must contain items")

def _required_prepared_artifact_paths(settings: CascadeSettings) -> list[Path]:
    root = Path(settings.artifacts_root)
    required = [
        root / "evidence_rows.jsonl",
        root / "stage1_pair_embeddings.npy",
        root / "stage1_pair_embeddings.meta.json",
        root / "stage2_pair_embeddings.npy",
        root / "stage2_pair_embeddings.meta.json",
    ]
    if settings.mode == "chunk":
        required.extend(
            [
                root / "selected_chunks.jsonl",
                root / "step1_batch_inputs.jsonl",
            ]
        )
    return required

def _load_artifact_meta(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except FileNotFoundError as exc:
        raise ArtifactContractError(f"missing required artifact meta: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(f"invalid artifact meta json: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"artifact meta must be a JSON object: {path}")
    return payload

def _meta_mismatches(
    settings: CascadeSettings,
    meta: dict[str, Any],
    *,
    embedding_stage: str,
) -> list[str]:
    expected_pairs: list[tuple[str, Any]] = [
        ("mode", str(settings.mode)),
        ("embedding_model", str(settings.embedding_model)),
        ("pair_embedding_version", str(PAIR_EMBEDDING_VERSION)),
        ("embedding_stage", str(embedding_stage)),
    ]
    mismatches: list[str] = []
    for key, expected in expected_pairs:
        actual = meta.get(key)
        if isinstance(expected, int):
            try:
                normalized_actual = int(actual)
            except (TypeError, ValueError):
                normalized_actual = None
            if normalized_actual != expected:
                mismatches.append(f"{key}: expected={expected!r} actual={actual!r}")
            continue
        normalized_actual = str(actual or "")
        if normalized_actual != expected:
            mismatches.append(f"{key}: expected={expected!r} actual={normalized_actual!r}")
    return mismatches

def validate_prepared_artifacts(settings: CascadeSettings) -> None:
    missing = [path for path in _required_prepared_artifact_paths(settings) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.name) for path in missing)
        raise ArtifactContractError(f"missing required artifacts: {missing_text}")
    mismatch_messages: list[str] = []
    for embedding_stage in ("stage1", "stage2"):
        meta_path = Path(settings.artifacts_root) / f"{embedding_stage}_pair_embeddings.meta.json"
        meta = _load_artifact_meta(meta_path)
        mismatches = _meta_mismatches(settings, meta, embedding_stage=embedding_stage)
        if mismatches:
            mismatch_messages.append(f"{meta_path.name}: " + "; ".join(mismatches))
    if mismatch_messages:
        raise ArtifactContractError("artifact meta mismatch: " + " | ".join(mismatch_messages))

def load_evidence_rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        return read_jsonl(path)
    except ValueError as exc:
        raise ArtifactContractError("evidence row must be a JSON object") from exc

def latest_rows_by_sample_id(root: Path, file_name: str, sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl_lenient(root / file_name):
        sample_id = str(row.get("sample_id", "") or "")
        if sample_id in sample_ids:
            latest[sample_id] = dict(row)
    return latest

def _source_from_value(value: Any) -> Source | None:
    try:
        return Source(str(value or ""))
    except ValueError:
        return None

def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _append_path_once(sample: Sample, value: str) -> None:
    if value and value not in sample.path:
        sample.path.append(value)

def filter_stage_rows_to_samples(root: Path, sample_ids: set[str]) -> None:
    if not root.exists():
        return
    jsonl_files = (
        "step1_rows.jsonl",
        "stage2_inference_rows.jsonl",
        "step2_rows.jsonl",
        "final_rows.jsonl",
    )
    for name in jsonl_files:
        path = root / name
        if path.exists():
            rows = [
                row
                for row in read_jsonl_lenient(path)
                if str(row.get("sample_id", "") or "") in sample_ids
            ]
            write_jsonl(path, rows)

def load_prepared_samples(settings: CascadeSettings) -> list[Sample]:
    validate_prepared_artifacts(settings)
    ground_truth_labels = load_ground_truth_index(settings.ground_truth or None)
    root = Path(settings.artifacts_root)
    rows = load_evidence_rows(root / "evidence_rows.jsonl")
    stage1_vectors = np.load(root / "stage1_pair_embeddings.npy", mmap_mode="r")
    stage2_vectors = np.load(root / "stage2_pair_embeddings.npy", mmap_mode="r")
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", "") or "")
        default_task_text = resolve_default_task_text(row)
        default_evidence_text = str(row.get("evidence_text", row.get("document_text", "")) or "")
        prompt_payload = resolve_prompt_payload(
            row,
            default_task_text=default_task_text,
            default_evidence_text=default_evidence_text,
        )
        embedding_payload = resolve_embedding_payload(
            row,
            default_task_text=default_task_text,
            default_evidence_text=default_evidence_text,
        )
        sample = Sample(
            sample_id=sample_id,
            text=str(row.get("document_text", "") or ""),
            query=str(row.get("query_text", "") or ""),
            label=normalize_binary_label(ground_truth_labels.get(sample_id, row.get("label", ""))) or None,
            raw_query_text=str(row.get("raw_query_text", row.get("query_text", "")) or ""),
            prompt_style=str(row.get("prompt_style", "query_document") or "query_document"),
            prompt_task_label=prompt_payload["task_label"],
            prompt_evidence_label=prompt_payload["evidence_label"],
            claim_text=str(row.get("claim_text", row.get("claim", "")) or ""),
            prompt_task_text=prompt_payload["task_text"],
            prompt_instruction_text=prompt_payload["instruction_text"],
            prompt_evidence_text=prompt_payload["evidence_text"],
            embedding_task_text=embedding_payload["task_text"],
            embedding_instruction_text=embedding_payload["instruction_text"],
            embedding_evidence_text=embedding_payload["evidence_text"],
        )
        stage2_prediction = normalize_binary_label(row.get("stage2_prediction", ""))
        if stage2_prediction in {"yes", "no"}:
            sample.stage2_prediction = stage2_prediction
        try:
            sample.stage2_routing_score = float(row.get("stage2_routing_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            sample.stage2_routing_score = 0.0
        if index < stage1_vectors.shape[0]:
            stage1_vector = np.asarray(stage1_vectors[index], dtype=np.float32)
            setattr(sample, "_stage1_pair_embedding", stage1_vector)
            setattr(sample, "_pair_embedding", stage1_vector)
        if index < stage2_vectors.shape[0]:
            setattr(sample, "_stage2_pair_embedding", np.asarray(stage2_vectors[index], dtype=np.float32))
        samples.append(sample)
    return samples

def step1_stage_row(sample: Sample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "prediction": normalize_binary_label(sample.stage1_prediction),
        "raw_label": str(sample.step1_raw_label or ""),
        "parse_failed": bool(sample.step1_parse_failed),
        "parse_failure_reason": str(sample.step1_parse_failure_reason or ""),
        "routing_score": float(sample.step1_routing_score),
        "signal_quality": str(sample.step1_signal_quality),
        "signal_source": str(sample.step1_signal_source),
        "yes_logprob": float(sample.step1_yes_logprob),
        "no_logprob": float(sample.step1_no_logprob),
        "paired_exact_prob": float(sample.step1_paired_exact_prob),
        "binary_entropy": float(sample.step1_binary_entropy),
    }

def hydrate_step1_from_rows(samples: list[Sample], root: Path) -> set[str]:
    sample_ids = {sample.sample_id for sample in samples}
    rows = latest_rows_by_sample_id(root, "step1_rows.jsonl", sample_ids)
    resumed: set[str] = set()
    for sample in samples:
        row = rows.get(sample.sample_id)
        if row is None:
            continue
        if "prediction" not in row and not bool(row.get("parse_failed", False)):
            continue
        prediction = normalize_binary_label(row.get("prediction", ""))
        sample.stage1_prediction = prediction if prediction in {"yes", "no"} else None
        sample.step1_raw_label = str(row.get("raw_label", row.get("prediction", "")) or "")
        sample.step1_parse_failed = bool(row.get("parse_failed", False))
        sample.step1_parse_failure_reason = str(row.get("parse_failure_reason", "") or "")
        sample.step1_routing_score = float(row.get("routing_score", 0.0) or 0.0)
        sample.step1_signal_quality = str(row.get("signal_quality", "") or "")
        sample.step1_signal_source = str(row.get("signal_source", "") or "")
        sample.step1_yes_logprob = float(row.get("yes_logprob", 0.0) or 0.0)
        sample.step1_no_logprob = float(row.get("no_logprob", 0.0) or 0.0)
        sample.step1_paired_exact_prob = float(row.get("paired_exact_prob", 0.0) or 0.0)
        sample.step1_binary_entropy = float(row.get("binary_entropy", 0.0) or 0.0)
        resumed.add(sample.sample_id)
    return resumed

def step2_stage_row(sample: Sample, route: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "decision": str(route.get("decision", "") or ""),
        "reason": str(route.get("reason", "") or ""),
        "pred": str(
            route.get("accepted_prediction")
            or route.get("pred", normalize_binary_label(sample.stage1_prediction))
            or ""
        ),
        "accepted_prediction": route.get("accepted_prediction"),
        "stage1_pred": route.get("stage1_pred"),
        "stage2_pred": route.get("stage2_pred"),
        "signal_quality": str(route.get("signal_quality", sample.step1_signal_quality) or ""),
        "routing_score": float(route.get("routing_score", sample.step1_routing_score) or 0.0),
        "stage2_routing_score": float(
            route.get("stage2_routing_score", sample.stage2_routing_score) or 0.0
        ),
        "threshold": route.get("decision_threshold"),
        "neighbor_support": route.get("neighbor_support"),
        "stage1_neighbor_support": route.get("stage1_neighbor_support"),
        "stage2_neighbor_support": route.get("stage2_neighbor_support"),
        "stage1_neighbor_rho": route.get("stage1_neighbor_rho"),
        "stage2_neighbor_rho": route.get("stage2_neighbor_rho"),
        "tilde_b_stage1": route.get("tilde_b_stage1"),
        "tilde_b_stage2": route.get("tilde_b_stage2"),
        "lambda_hat": route.get("lambda_hat"),
        "lambda1_hat": route.get("lambda1_hat"),
        "lambda2_hat": route.get("lambda2_hat"),
        "decision_stage": route.get("decision_stage"),
        "stage1_accept": route.get("stage1_accept"),
        "stage2_accept": route.get("stage2_accept"),
        "in_I2_max": route.get("in_I2_max"),
        "stage2_prediction_available": route.get("stage2_prediction_available"),
        "stage1_threshold": route.get("stage1_decision_threshold"),
        "stage2_threshold": route.get("stage2_decision_threshold"),
    }

def hydrate_step2_from_rows(samples: list[Sample], root: Path) -> set[str]:
    sample_ids = {sample.sample_id for sample in samples}
    rows = latest_rows_by_sample_id(root, "step2_rows.jsonl", sample_ids)
    resumed: set[str] = set()
    for sample in samples:
        row = rows.get(sample.sample_id)
        if row is None:
            continue
        decision = str(row.get("decision", "") or "").strip().lower()
        if decision not in {"accept", "defer"}:
            continue
        pred = (
            normalize_binary_label(row.get("accepted_prediction", ""))
            or normalize_binary_label(row.get("pred", ""))
            or normalize_binary_label(sample.stage1_prediction)
        )
        support_value = float(row.get("neighbor_support", sample.step2_neighbor_support) or 0.0)
        sample.step2_neighbor_support = support_value
        sample.step2_decision_threshold_used = _optional_float(row.get("threshold"))
        sample.step2_lambda_hat_used = _optional_float(row.get("lambda_hat"))
        if decision == "accept":
            sample.action = Action.ACCEPT
            sample.prediction = pred
            sample.source = (
                Source.STEP2_LOCAL
                if str(row.get("decision_stage", "") or "") == "stage2"
                else Source.STEP1_LOCAL
            )
            sample.defer_reason = ""
            _append_path_once(sample, "step2:accept")
        else:
            sample.action = Action.DEFER
            sample.defer_reason = str(row.get("reason", "step2_defer") or "step2_defer")
            _append_path_once(sample, f"step2:defer(reason={sample.defer_reason})")
        resumed.add(sample.sample_id)
    return resumed

def hydrate_final_rows(samples: list[Sample], root: Path) -> int:
    sample_ids = {sample.sample_id for sample in samples}
    rows = latest_rows_by_sample_id(root, "final_rows.jsonl", sample_ids)
    resumed = 0
    for sample in samples:
        row = rows.get(sample.sample_id)
        if row is None:
            continue
        prediction = normalize_binary_label(row.get("prediction", ""))
        if prediction not in {"yes", "no"}:
            continue
        sample.prediction = prediction
        sample.action = Action.ACCEPT
        source = _source_from_value(row.get("source", ""))
        if source is not None:
            sample.source = source
        sample.path = list(row.get("path", []) or [])
        sample.defer_reason = str(row.get("defer_reason", "") or "")
        resumed += 1
    return resumed

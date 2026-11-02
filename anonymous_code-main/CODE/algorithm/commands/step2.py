from __future__ import annotations
from argparse import Namespace
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from algorithm.commands.step1 import LocalModelManager
from algorithm.core.step2_wsr import Step2WSR
from algorithm.settings import CascadeSettings
from algorithm.common.artifacts import (
    filter_stage_rows_to_samples,
    hydrate_final_rows,
    hydrate_step1_from_rows,
    latest_rows_by_sample_id,
    load_prepared_samples,
    step2_stage_row,
)
from algorithm.common.metrics import select_calibration_samples_srswor
from algorithm.common.runtime import Action, Sample, Source
from script.io import write_json
from script.model_usage import build_cost_summary, elapsed_seconds_since, monotonic_seconds
from script.labels import normalize_binary_label
from script.stream import StreamingRunWriter
from script.vllm import get_local_vllm_gpu_profile, parallel_map_ordered
_DEFAULT_CALIBRATION_SAMPLE_SEED = 42
_DEFAULT_STAGE2_DOCUMENT_PARALLEL_REQUESTS = 8

def _settings_from_args(args: Namespace) -> CascadeSettings:
    return CascadeSettings(
        command="step2",
        mode=str(args.mode),
        queries=str(args.queries),
        documents=str(args.documents),
        ground_truth=str(getattr(args, "ground_truth", "") or ""),
        artifacts_root=str(args.artifacts_root),
        embedding_model=str(args.embedding_model),
        step2_chat_model=str(getattr(args, "step2_chat_model", "") or ""),
        step2_chat_base_url=str(getattr(args, "step2_chat_base_url", "") or ""),
        step2_chat_local_path=str(getattr(args, "step2_chat_local_path", "") or ""),
        step2_max_parallel_requests=int(getattr(args, "step2_max_parallel_requests", 0) or 0),
        tau=float(getattr(args, "tau", 0.95)),
        delta_w=float(getattr(args, "delta_w", 0.05)),
        combiner_source=str(getattr(args, "combiner_source", "source1") or "source1"),
        combiner_theta_grid=str(getattr(args, "combiner_theta_grid", "") or ""),
    )

def _parse_combiner_theta_grid(raw_value: str) -> list[dict[str, Any]] | None:
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    payload = json.loads(raw)
    if isinstance(payload, dict) and isinstance(payload.get("theta_grid"), list):
        payload = payload["theta_grid"]
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError("combiner_theta_grid must be a JSON list of objects")
    return [dict(item) for item in payload]

@dataclass(frozen=True)
class _Stage2Config:
    step1_chat_base_url: str
    step1_chat_model_name: str
    step1_chat_timeout: int = 120
    step1_max_parallel_requests: int = _DEFAULT_STAGE2_DOCUMENT_PARALLEL_REQUESTS
    step1_server_max_num_seqs: int = 256
    step1_server_gpu_memory_utilization: float = 0.9
    step1_chat_local_path: str = ""

    @classmethod
    def from_settings(cls, settings: CascadeSettings) -> "_Stage2Config":
        gpu_profile = get_local_vllm_gpu_profile() or {}
        profile_parallel_cap = int(gpu_profile.get("chat_parallel_requests", 0) or 0)
        requested_parallel = int(settings.step2_max_parallel_requests or 0)
        parallel_requests = (
            requested_parallel
            if requested_parallel > 0
            else (
                profile_parallel_cap
                if profile_parallel_cap > 0
                else _DEFAULT_STAGE2_DOCUMENT_PARALLEL_REQUESTS
            )
        )
        return cls(
            step1_chat_base_url=str(settings.step2_chat_base_url),
            step1_chat_model_name=str(settings.step2_chat_model),
            step1_max_parallel_requests=max(1, int(parallel_requests)),
            step1_server_max_num_seqs=max(1, int(gpu_profile.get("chat_max_num_seqs", 256) or 256)),
            step1_server_gpu_memory_utilization=float(
                gpu_profile.get("gpu_memory_utilization", 0.90) or 0.90
            ),
            step1_chat_local_path=str(settings.step2_chat_local_path or ""),
        )

def _empty_stage2_inference_usage(*, enabled: bool = False, resumed_count: int = 0) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "llm_calls_total": 0,
        "prompt_tokens_total": 0,
        "system_prompt_tokens_total": 0,
        "user_prompt_tokens_total": 0,
        "completion_tokens_total": 0,
        "total_tokens_total": 0,
        "batch_request_count": 0,
        "document_count": 0,
        "failed_document_count": 0,
        "resumed_count": int(resumed_count),
        "token_accounting_mode": "provider_usage",
    }

def _add_stage2_usage(total: dict[str, Any], result: dict[str, Any]) -> None:
    usage = dict(result.get("usage", {}) or {})
    total["llm_calls_total"] = int(total.get("llm_calls_total", 0) or 0) + 1
    total["batch_request_count"] = int(total.get("batch_request_count", 0) or 0) + 1
    total["document_count"] = int(total.get("document_count", 0) or 0) + 1
    for source_key, total_key in (
        ("prompt_tokens", "prompt_tokens_total"),
        ("system_prompt_tokens", "system_prompt_tokens_total"),
        ("user_prompt_tokens", "user_prompt_tokens_total"),
        ("completion_tokens", "completion_tokens_total"),
        ("total_tokens", "total_tokens_total"),
    ):
        total[total_key] = int(total.get(total_key, 0) or 0) + int(usage.get(source_key, 0) or 0)

def _stage2_prediction_available(sample: Sample) -> bool:
    return normalize_binary_label(sample.stage2_prediction) in {"yes", "no"}

def _stage2_inference_target(sample: Sample) -> bool:
    # Algorithm 1 treats an unusable/malformed Stage-1 output as a defer,
    # so it must still be eligible for the full-document Stage-2 pass.
    return not _stage2_prediction_available(sample)

def _hydrate_stage2_inference_from_rows(samples: list[Sample], root: Path) -> set[str]:
    sample_ids = {sample.sample_id for sample in samples}
    rows = latest_rows_by_sample_id(root, "stage2_inference_rows.jsonl", sample_ids)
    resumed: set[str] = set()
    for sample in samples:
        row = rows.get(sample.sample_id)
        if row is None:
            continue
        prediction = normalize_binary_label(row.get("stage2_prediction", ""))
        if prediction not in {"yes", "no"}:
            continue
        sample.stage2_prediction = prediction
        try:
            sample.stage2_routing_score = float(row.get("stage2_routing_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            sample.stage2_routing_score = 0.0
        resumed.add(sample.sample_id)
    return resumed

def _stage2_inference_row(sample: Sample, result: dict[str, Any]) -> dict[str, Any]:
    usage = dict(result.get("usage", {}) or {})
    return {
        "sample_id": sample.sample_id,
        "stage2_prediction": normalize_binary_label(sample.stage2_prediction),
        "stage2_routing_score": float(sample.stage2_routing_score or 0.0),
        "raw_label": str(result.get("raw_label", "") or ""),
        "parse_failed": bool(result.get("parse_failed", False)),
        "parse_failure_reason": str(result.get("parse_failure_reason", "") or ""),
        "signal_quality": str(result.get("step1_signal_quality", "") or ""),
        "signal_source": str(result.get("step1_signal_source", "") or ""),
        "yes_logprob": float(result.get("step1_yes_logprob", 0.0) or 0.0),
        "no_logprob": float(result.get("step1_no_logprob", 0.0) or 0.0),
        "paired_exact_prob": float(result.get("step1_paired_exact_prob", 0.0) or 0.0),
        "binary_entropy": float(result.get("step1_binary_entropy", 0.0) or 0.0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }

def _apply_stage2_inference_result(sample: Sample, result: dict[str, Any]) -> None:
    prediction = normalize_binary_label(result.get("answer", ""))
    if prediction in {"yes", "no"}:
        sample.stage2_prediction = prediction
        sample.stage2_routing_score = float(result.get("step1_routing_score", 0.0) or 0.0)
    else:
        sample.stage2_prediction = None
        sample.stage2_routing_score = 0.0
    sample.latency_ms += float(result.get("latency_ms", 0.0) or 0.0)

def _stage2_inference_exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "answer": None,
        "raw_label": "",
        "step1_yes_logprob": 0.0,
        "step1_no_logprob": 0.0,
        "step1_paired_exact_prob": 0.0,
        "step1_signal_quality": "parse_failed",
        "step1_signal_source": "stage2_inference_exception",
        "step1_routing_score": 0.0,
        "step1_binary_entropy": 0.0,
        "latency_ms": 0.0,
        "usage": {},
        "parse_failed": True,
        "parse_failure_reason": str(exc) or exc.__class__.__name__,
    }

def _run_stage2_inference(
    settings: CascadeSettings,
    samples: list[Sample],
    *,
    writer: StreamingRunWriter,
) -> dict[str, Any]:
    resumed_ids = _hydrate_stage2_inference_from_rows(samples, writer.root)
    if not str(settings.step2_chat_model or "").strip():
        return _empty_stage2_inference_usage(enabled=False, resumed_count=len(resumed_ids))
    targets = [sample for sample in samples if sample.sample_id not in resumed_ids and _stage2_inference_target(sample)]
    usage = _empty_stage2_inference_usage(enabled=True, resumed_count=len(resumed_ids))
    if not targets:
        return usage
    config = _Stage2Config.from_settings(settings)
    manager = LocalModelManager(config)
    parallel = max(1, int(config.step1_max_parallel_requests or 1))
    processed = 0

    def _infer_one(sample: Sample) -> dict[str, Any]:
        return manager._forward_vllm_single_token(
            str(sample.text or ""),
            str(sample.query or ""),
            task_label=str(sample.prompt_task_label or "Query"),
            evidence_label=str(sample.prompt_evidence_label or "Document"),
            task_text=str(sample.prompt_task_text or sample.claim_text or sample.query or ""),
            instruction_text=str(sample.prompt_instruction_text or ""),
        )

    def _flush(sample: Sample, result: dict[str, Any]) -> None:
        nonlocal processed
        _apply_stage2_inference_result(sample, result)
        if isinstance(result.get("usage", {}), dict) and result.get("usage"):
            _add_stage2_usage(usage, result)
        if bool(result.get("parse_failed", False)):
            usage["failed_document_count"] = int(usage.get("failed_document_count", 0) or 0) + 1
        writer.write_stage_row("stage2_inference_rows", _stage2_inference_row(sample, result))
        processed += 1
        writer.write_progress(
            stage="stage2_inference",
            completed=processed + len(resumed_ids),
            total=len(targets) + len(resumed_ids),
            mode=settings.mode,
            resumed=len(resumed_ids),
        )

    def _run_batch(batch: list[Sample]) -> None:
        if not batch:
            return
        try:
            if parallel > 1 and len(batch) > 1:
                results = parallel_map_ordered(items=batch, fn=_infer_one, max_workers=parallel)
            else:
                results = [_infer_one(sample) for sample in batch]
        except Exception as exc:
            if len(batch) <= 1:
                _flush(batch[0], _stage2_inference_exception_result(exc))
                return
            midpoint = max(1, len(batch) // 2)
            writer.write_event(
                "stage2_inference_batch_split_after_error",
                mode=settings.mode,
                batch_size=len(batch),
                left_size=midpoint,
                right_size=len(batch) - midpoint,
                error_type=exc.__class__.__name__,
                error=str(exc)[:500],
            )
            _run_batch(batch[:midpoint])
            _run_batch(batch[midpoint:])
            return
        for sample, result in zip(batch, results, strict=True):
            _flush(sample, result)

    writer.write_event(
        "stage2_inference_started",
        mode=settings.mode,
        sample_count=len(targets),
        resumed_count=len(resumed_ids),
    )
    for start in range(0, len(targets), parallel):
        _run_batch(targets[start : start + parallel])
    return usage

def _calibrate_step2(
    model: Step2WSR,
    samples: list[Any],
    *,
    calibration_ratio: float,
    sample_seed: int,
) -> dict[str, Any]:
    if calibration_ratio <= 0.0:
        return {
            "requested_ratio": float(calibration_ratio),
            "applied": False,
            "reason": "disabled",
            "sample_count": 0,
            "snapshot": model.snapshot(),
        }
    selected = select_calibration_samples_srswor(
        samples,
        calibration_ratio=float(calibration_ratio),
        seed=int(sample_seed),
    )
    labeled = [sample for sample in selected if normalize_binary_label(sample.label) in {"yes", "no"}]
    model.calibrate(labeled, neighbor_support_samples=samples, population_size=len(samples))
    return {
        "requested_ratio": float(calibration_ratio),
        "applied": bool(labeled),
        "reason": "applied" if labeled else "no_labeled_samples",
        "sample_count": int(len(labeled)),
        "snapshot": model.snapshot(),
    }

def run_from_args(args: Namespace) -> dict[str, Any]:
    started_at = monotonic_seconds()
    settings = _settings_from_args(args)
    samples = load_prepared_samples(settings)
    output_dir = Path(settings.artifacts_root) / "stage_output"
    sample_ids = {sample.sample_id for sample in samples}
    filter_stage_rows_to_samples(output_dir, sample_ids)
    resumed_final_count = hydrate_final_rows(samples, output_dir)
    writer = StreamingRunWriter(output_dir)
    writer.write_event("step2_stage_started", mode=settings.mode, resumed_final_count=resumed_final_count)
    candidates = [sample for sample in samples if sample.action != Action.ACCEPT]
    resumed_step1_ids = hydrate_step1_from_rows(candidates, writer.root)
    missing_step1_ids = sorted(
        sample.sample_id
        for sample in candidates
        if sample.sample_id not in resumed_step1_ids
    )
    if missing_step1_ids:
        raise RuntimeError(
            "step2 requires cached step1 rows for samples: "
            + ", ".join(missing_step1_ids[:10])
        )
    stage2_inference_usage = _run_stage2_inference(settings, candidates, writer=writer)
    model = Step2WSR(
        tau=float(settings.tau),
        delta_w=float(settings.delta_w),
        embedding_model=str(settings.embedding_model),
        combiner_source=str(settings.combiner_source),
        combiner_theta_grid=_parse_combiner_theta_grid(settings.combiner_theta_grid),
    )
    calibration = _calibrate_step2(
        model,
        candidates,
        calibration_ratio=float(getattr(args, "calibration_ratio", 0.0) or 0.0),
        sample_seed=_DEFAULT_CALIBRATION_SAMPLE_SEED,
    )
    writer.write_event(
        "calibration_finished",
        applied=bool(calibration.get("applied")),
        sample_count=int(calibration.get("sample_count", 0) or 0),
    )
    pending = [sample for sample in candidates if sample.action != Action.ACCEPT]
    model.annotate_neighbor_support(pending, exclude_self=True)
    for sample in pending:
        pred = normalize_binary_label(sample.stage1_prediction)
        if sample.step1_parse_failed or pred not in {"yes", "no"}:
            route = {
                "decision": "defer",
                "reason": "step1_parse_failed",
                "pred": pred,
                "signal_quality": sample.step1_signal_quality,
                "routing_score": sample.step1_routing_score,
                "neighbor_support": sample.step2_neighbor_support,
                "stage1_neighbor_support": sample.step1_neighbor_support,
                "stage2_neighbor_support": sample.step2_neighbor_support,
                "stage1_neighbor_rho": sample.step1_neighbor_rho,
                "stage2_neighbor_rho": sample.step2_neighbor_rho,
                "tilde_b_stage1": sample.step1_tilde_b,
                "tilde_b_stage2": sample.step2_tilde_b,
                "decision_threshold": None,
                "lambda_hat": None,
            }
            sample.action = Action.DEFER
            sample.defer_reason = "step1_parse_failed"
        else:
            route = model.decide(sample)
            sample.step2_decision_threshold_used = route.get("decision_threshold")
            sample.step2_lambda_hat_used = route.get("lambda_hat")
            if route.get("decision") == "accept":
                sample.action = Action.ACCEPT
                accepted_prediction = normalize_binary_label(
                    route.get("accepted_prediction", "")
                ) or pred
                sample.prediction = accepted_prediction
                sample.source = (
                    Source.STEP2_LOCAL
                    if str(route.get("decision_stage", "")) == "stage2"
                    else Source.STEP1_LOCAL
                )
                sample.defer_reason = ""
                sample.path.append(f"step2:accept(stage={route.get('decision_stage', '') or 'unknown'})")
            else:
                sample.action = Action.DEFER
                sample.defer_reason = str(route.get("reason", "step2_defer") or "step2_defer")
                sample.path.append(f"step2:defer(reason={sample.defer_reason})")
        writer.write_stage_row("step2_rows", step2_stage_row(sample, route))
    elapsed_seconds = elapsed_seconds_since(started_at)
    model_call_count = int(stage2_inference_usage.get("llm_calls_total", 0) or 0)
    cost = build_cost_summary(
        elapsed_seconds=elapsed_seconds,
        model_call_count=model_call_count,
        model_request_count=int(stage2_inference_usage.get("batch_request_count", model_call_count) or 0),
        model_input_count=int(stage2_inference_usage.get("document_count", model_call_count) or 0),
        prompt_tokens=int(stage2_inference_usage.get("prompt_tokens_total", 0) or 0),
        system_prompt_tokens=int(stage2_inference_usage.get("system_prompt_tokens_total", 0) or 0),
        user_prompt_tokens=int(stage2_inference_usage.get("user_prompt_tokens_total", 0) or 0),
        completion_tokens=int(stage2_inference_usage.get("completion_tokens_total", 0) or 0),
    )
    payload = {
        "stage": "step2",
        "mode": str(settings.mode),
        "total_samples": int(len(samples)),
        "cached_final_count": int(resumed_final_count),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "stage2_inference_usage": stage2_inference_usage,
        "calibration": calibration,
        "step2_snapshot": model.snapshot(),
    }
    writer.write_summary(payload)
    write_json(output_dir / f"step2_{settings.mode}.json", payload)
    return payload
__all__ = ["run_from_args"]

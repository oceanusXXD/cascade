from __future__ import annotations
from argparse import Namespace
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from algorithm.settings import CascadeSettings, DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE
from algorithm.common.artifacts import (
    filter_stage_rows_to_samples,
    hydrate_final_rows,
    hydrate_step1_from_rows,
    load_prepared_samples,
    step1_stage_row,
)
from script.io import read_jsonl, write_json
from script.model_usage import (
    build_cost_summary,
    elapsed_seconds_since,
    estimate_prompt_token_breakdown,
    monotonic_seconds,
)
from script.stream import StreamingRunWriter
from script.vllm import (
    create_chat_completion as _create_chat_completion,
    ensure_local_vllm_chat_ready as _ensure_local_vllm_chat_ready,
    extract_response_usage,
    get_local_vllm_gpu_profile,
    get_openai_compatible_client,
    parallel_map_ordered as _parallel_map_ordered,
    vllm_chat_result,
)
from algorithm.common.runtime import (
    Action,
    apply_step1_result,
    Sample,
    build_step1_prompt,
    build_step1_system_prompt,
)
from algorithm.core.step1 import (
    build_chunk_step1_system_prompt,
    create_chunk_step1_prompt,
    extract_chunk_step1_label_scores,
    normalize_chunk_step1_score_row,
    parse_chunk_step1_output,
)
from script.labels import normalize_binary_decision_text
logger = logging.getLogger(__name__)
_DEFAULT_STEP1_DOCUMENT_PARALLEL_REQUESTS = 8
_DEFAULT_STEP1_BATCH_TOKEN_BUDGET = 4096


def _resolve_chunk_step1_batch_token_budget() -> int:
    raw = str(os.environ.get("CASCADE_STEP1_BATCH_TOKEN_BUDGET", "") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass
    gpu_profile = get_local_vllm_gpu_profile() or {}
    profile_budget = int(gpu_profile.get("chat_max_model_len", 0) or 0)
    if profile_budget > 0:
        return profile_budget
    return _DEFAULT_STEP1_BATCH_TOKEN_BUDGET


def _chunk_step1_prompt_token_count(batch_payload: dict[str, Any], items: list[dict[str, Any]]) -> int:
    prompt = create_chunk_step1_prompt(
        str(batch_payload.get("query_text", "") or ""),
        items,
        task_label=str(batch_payload.get("prompt_task_label", "Query") or "Query"),
        evidence_label=str(batch_payload.get("prompt_evidence_label", "Document") or "Document"),
        instruction_text=str(batch_payload.get("prompt_instruction_text", "") or ""),
    )
    system_prompt = build_chunk_step1_system_prompt(
        task_label=str(batch_payload.get("prompt_task_label", "Query") or "Query"),
        evidence_label=str(batch_payload.get("prompt_evidence_label", "Document") or "Document"),
        instruction_text=str(batch_payload.get("prompt_instruction_text", "") or ""),
    )
    breakdown = estimate_prompt_token_breakdown(system_texts=[system_prompt], user_texts=[prompt])
    return max(1, int(breakdown.get("prompt_tokens", 0) or 0))


def _rebatch_chunk_step1_payloads(
    prepared_batches: list[dict[str, Any]],
    *,
    pending_ids: set[str],
    token_budget: int,
) -> list[dict[str, Any]]:
    """Pack Stage-1 items greedily subject to the paper's B_max token budget."""
    grouped: dict[str, dict[str, Any]] = {}
    seen_sample_ids: set[str] = set()
    for prepared in prepared_batches:
        query_id = str(prepared.get("query_id", "") or "")
        group = grouped.setdefault(
            query_id,
            {
                "base": dict(prepared),
                "items": [],
            },
        )
        for raw_item in list(prepared.get("items", []) or []):
            if not isinstance(raw_item, dict):
                continue
            sample_id = str(raw_item.get("sample_id", "") or "")
            if sample_id not in pending_ids or sample_id in seen_sample_ids:
                continue
            seen_sample_ids.add(sample_id)
            group["items"].append(dict(raw_item))

    packed: list[dict[str, Any]] = []
    resolved_budget = max(1, int(token_budget))
    for query_id, group in grouped.items():
        base = dict(group["base"])
        items = list(group["items"])
        batch_index = 0

        def _emit(selected: list[dict[str, Any]]) -> None:
            nonlocal batch_index
            if not selected:
                return
            batch_index += 1
            normalized_items = []
            for display_index, item in enumerate(selected, start=1):
                normalized = dict(item)
                normalized["display_index"] = int(display_index)
                normalized_items.append(normalized)
            payload = dict(base)
            payload["batch_id"] = f"{query_id}:token:{batch_index:04d}"
            payload["sample_ids"] = [str(item.get("sample_id", "") or "") for item in normalized_items]
            payload["items"] = normalized_items
            packed.append(payload)

        current: list[dict[str, Any]] = []
        for item in items:
            candidate = [*current, item]
            candidate_tokens = _chunk_step1_prompt_token_count(base, candidate)
            if current and candidate_tokens > resolved_budget:
                _emit(current)
                current = [item]
            else:
                current = candidate
        _emit(current)
    return packed

def _resolve_step1_parallel_requests(settings: CascadeSettings) -> tuple[dict[str, Any], int]:
    gpu_profile = get_local_vllm_gpu_profile() or {}
    profile_parallel_cap = int(gpu_profile.get("chat_parallel_requests", 0) or 0)
    requested_parallel = int(settings.step1_max_parallel_requests or 0)
    if requested_parallel <= 0:
        parallel_requests = (
            profile_parallel_cap
            if profile_parallel_cap > 0
            else _DEFAULT_STEP1_DOCUMENT_PARALLEL_REQUESTS
        )
    else:
        parallel_requests = requested_parallel
    return gpu_profile, max(1, int(parallel_requests))

@dataclass(frozen=True)

class _Step1Config:
    queries_file: str
    documents_file: str
    step1_chat_base_url: str
    step1_chat_model_name: str
    step1_chat_timeout: int = 120
    step1_max_parallel_requests: int = _DEFAULT_STEP1_DOCUMENT_PARALLEL_REQUESTS
    step1_server_max_num_seqs: int = 256
    step1_server_gpu_memory_utilization: float = 0.9
    step1_chat_local_path: str = ""
    @classmethod
    def from_settings(cls, settings: CascadeSettings) -> "_Step1Config":
        gpu_profile, parallel_requests = _resolve_step1_parallel_requests(settings)
        return cls(
            queries_file=str(settings.queries),
            documents_file=str(settings.documents),
            step1_chat_base_url=str(settings.step1_chat_base_url),
            step1_chat_model_name=str(settings.step1_chat_model),
            step1_max_parallel_requests=max(1, int(parallel_requests)),
            step1_server_max_num_seqs=max(1, int(gpu_profile.get("chat_max_num_seqs", 256) or 256)),
            step1_server_gpu_memory_utilization=float(
                gpu_profile.get("gpu_memory_utilization", 0.90) or 0.90
            ),
            step1_chat_local_path=str(settings.step1_chat_local_path or ""),
        )

class LocalModelManager:
    def __init__(self, config: Any):
        self.config = config
        self._vllm_client = None
        self._vllm_model_name = str(getattr(config, "step1_chat_model_name", "") or "").strip()
    def load(self) -> None:
        if self._vllm_client is not None:
            return
        if not self._vllm_model_name:
            raise RuntimeError("step1_chat_model_name is required for step1 logit inference")
        self._vllm_client = _ensure_local_vllm_chat_ready(
            {
                "scope": "cascade_step1",
                "model": self._vllm_model_name,
                "api_key": "EMPTY",
                "base_url": str(self.config.step1_chat_base_url),
                "timeout": int(self.config.step1_chat_timeout),
                "local_model_path": str(getattr(self.config, "step1_chat_local_path", "") or "").strip(),
                "max_num_seqs": int(getattr(self.config, "step1_server_max_num_seqs", 256) or 256),
                "gpu_memory_utilization": float(
                    getattr(self.config, "step1_server_gpu_memory_utilization", 0.90) or 0.90
                ),
            }
        )
        logger.info(
            "[algorithm step1] local runtime ready: model=%s",
            self._vllm_model_name,
        )
    @staticmethod
    def _parse_binary_answer(text: str) -> Tuple[Optional[str], str, str]:
        raw = str(text or "")
        stripped = raw.strip()
        if not stripped:
            return None, "", "empty_output"
        compact = re.sub(r"\s+", "", stripped)
        for candidate in (stripped, compact):
            answer = normalize_binary_decision_text(candidate)
            if answer in {"yes", "no"}:
                raw_label = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", str(candidate).strip())
                return answer, raw_label, ""
        if re.search(r"\b(?:[Yy]|[Nn]|[Yy][Ee][Ss]|[Nn][Oo])\b", stripped):
            return None, "", "non_strict_binary_token"
        return None, "", "unrecognized_output"
    @staticmethod
    def _normalize_logprob_token(token: str) -> str:
        raw = str(token or "")
        stripped = raw.strip()
        upper = stripped.upper()
        if upper in {"Y", "N"}:
            return upper
        edge_cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", stripped).upper()
        if edge_cleaned in {"Y", "N"}:
            return edge_cleaned
        alpha_only = re.sub(r"[^A-Za-z]", "", stripped).upper()
        if alpha_only in {"Y", "N"} and (len(stripped) - len(alpha_only)) <= 2:
            return alpha_only
        return ""
    @classmethod
    def _collect_binary_logprobs_from_toplogprobs(cls, top_logprobs: Any) -> Dict[str, float]:
        observed: Dict[str, float] = {}
        if not top_logprobs:
            return observed
        for item in top_logprobs:
            value = getattr(item, "logprob", None)
            if value is None:
                continue
            token_norm = cls._normalize_logprob_token(getattr(item, "token", ""))
            if token_norm not in {"Y", "N"}:
                continue
            lp = float(value)
            prev = observed.get(token_norm)
            if prev is None or lp > prev:
                observed[token_norm] = lp
        return observed
    @staticmethod
    def _iter_candidate_logprob_positions(logprobs_obj: Any, max_positions: int = 8):
        if logprobs_obj is None:
            return
        content_items = getattr(logprobs_obj, "content", None)
        if not content_items:
            return
        for item in list(content_items)[: max(1, int(max_positions))]:
            top_logprobs = getattr(item, "top_logprobs", None)
            if top_logprobs:
                yield top_logprobs
    @classmethod
    def _inspect_step1_signal_from_logprobs(
        cls,
        logprobs_obj: Any,
        answer: str,
        max_positions: int = 8,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "signal_source": "logprobs_unavailable",
            "yes_logprob": None,
            "no_logprob": None,
            "paired_exact_prob": None,
            "routing_score": None,
            "binary_entropy": None,
            "signal_quality": "unstable_split",
        }
        if answer not in {"yes", "no"}:
            info["signal_source"] = "non_binary_answer"
            info["signal_quality"] = "parse_failed"
            return info
        if logprobs_obj is None:
            return info
        best_y_lp: Optional[float] = None
        best_n_lp: Optional[float] = None
        first_single_pred_lp: Optional[float] = None
        first_single_yes_lp: Optional[float] = None
        first_single_no_lp: Optional[float] = None
        saw_position = False
        saw_paired_sign_mismatch = False
        for top_logprobs in cls._iter_candidate_logprob_positions(logprobs_obj, max_positions=max_positions):
            saw_position = True
            observed = cls._collect_binary_logprobs_from_toplogprobs(top_logprobs)
            y_lp = observed.get("Y")
            n_lp = observed.get("N")
            pred_lp = y_lp if answer == "yes" else n_lp
            opp_lp = n_lp if answer == "yes" else y_lp
            if y_lp is not None and (best_y_lp is None or y_lp > best_y_lp):
                best_y_lp = y_lp
            if n_lp is not None and (best_n_lp is None or n_lp > best_n_lp):
                best_n_lp = n_lp
            if y_lp is not None and n_lp is not None:
                prob_yes = float(1.0 / (1.0 + math.exp(float(n_lp - y_lp))))
                prob_no = float(1.0 - prob_yes)
                pred_prob = float(prob_yes if answer == "yes" else prob_no)
                entropy = float(
                    -(
                        prob_yes * math.log(max(prob_yes, 1e-12))
                        + prob_no * math.log(max(prob_no, 1e-12))
                    )
                )
                logprob_gap = float((y_lp - n_lp) if answer == "yes" else (n_lp - y_lp))
                if logprob_gap >= 0.0:
                    info["signal_source"] = "paired_y_n_top_logprobs"
                    info["yes_logprob"] = float(y_lp)
                    info["no_logprob"] = float(n_lp)
                    info["paired_exact_prob"] = float(pred_prob)
                    info["routing_score"] = float(pred_prob)
                    info["binary_entropy"] = float(entropy)
                    info["signal_quality"] = "paired_exact"
                    return info
                saw_paired_sign_mismatch = True
                continue
            if pred_lp is not None and opp_lp is None and first_single_pred_lp is None:
                first_single_pred_lp = float(pred_lp)
                first_single_yes_lp = float(y_lp) if y_lp is not None else None
                first_single_no_lp = float(n_lp) if n_lp is not None else None
        if not saw_position:
            info["signal_source"] = "logprobs_unavailable"
            return info
        info["yes_logprob"] = (
            first_single_yes_lp
            if first_single_yes_lp is not None
            else (float(best_y_lp) if best_y_lp is not None else None)
        )
        info["no_logprob"] = (
            first_single_no_lp
            if first_single_no_lp is not None
            else (float(best_n_lp) if best_n_lp is not None else None)
        )
        if first_single_pred_lp is not None:
            info["signal_source"] = "single_sided_lower_bound"
            info["signal_quality"] = "single_sided_lb"
            info["routing_score"] = float(max(0.0, min(1.0, math.exp(float(first_single_pred_lp)))))
            return info
        if saw_paired_sign_mismatch:
            info["signal_source"] = "paired_y_n_sign_mismatch"
            return info
        if best_y_lp is None and best_n_lp is None:
            info["signal_source"] = "missing_candidates"
        elif (answer == "yes" and best_y_lp is None) or (answer == "no" and best_n_lp is None):
            info["signal_source"] = "missing_predicted_candidate"
        else:
            info["signal_source"] = "split_candidates_ignored"
        return info
    def _forward_vllm_single_token(
        self,
        text: str,
        query: str,
        *,
        task_label: str = "Query",
        evidence_label: str = "Document",
        task_text: str = "",
        instruction_text: str = "",
    ) -> Dict[str, Any]:
        if self._vllm_client is None:
            self.load()
        prompt = build_step1_prompt(
            text,
            query,
            task_label=task_label,
            evidence_label=evidence_label,
            task_text=task_text,
            instruction_text=instruction_text,
        )
        system_prompt = build_step1_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        start = time.time()
        response = _create_chat_completion(
            client=self._vllm_client,
            model=self._vllm_model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=1,
            disable_thinking=True,
            logprobs=True,
            top_logprobs=8,
        )
        latency_ms = (time.time() - start) * 1000.0
        usage = extract_response_usage(response)
        prompt_breakdown = estimate_prompt_token_breakdown(system_texts=[system_prompt], user_texts=[prompt])
        usage["system_prompt_tokens"] = int(prompt_breakdown["system_prompt_tokens"])
        usage["user_prompt_tokens"] = int(prompt_breakdown["user_prompt_tokens"])
        if not int(usage.get("prompt_tokens", 0) or 0):
            usage["prompt_tokens"] = int(prompt_breakdown["prompt_tokens"])
        choice = response.choices[0]
        content = choice.message.content
        answer, raw_label, parse_failure_reason = self._parse_binary_answer(str(content or ""))
        if answer not in {"yes", "no"}:
            return {
                "answer": None,
                "raw_label": raw_label,
                "step1_yes_logprob": 0.0,
                "step1_no_logprob": 0.0,
                "step1_paired_exact_prob": 0.0,
                "step1_signal_quality": "parse_failed",
                "step1_signal_source": "parse_failed",
                "step1_routing_score": 0.0,
                "step1_binary_entropy": 0.0,
                "latency_ms": latency_ms,
                "usage": usage,
                "parse_failed": True,
                "parse_failure_reason": str(parse_failure_reason or "unrecognized_output"),
            }
        logprobs_obj = getattr(choice, "logprobs", None)
        signal_info = self._inspect_step1_signal_from_logprobs(logprobs_obj, answer)
        signal_quality = str(signal_info.get("signal_quality", "unstable_split") or "unstable_split")
        routing_score = float(signal_info.get("routing_score", 0.0) or 0.0)
        if signal_quality != "paired_exact":
            return {
                "answer": answer,
                "raw_label": raw_label,
                "step1_yes_logprob": float(signal_info.get("yes_logprob", 0.0) or 0.0),
                "step1_no_logprob": float(signal_info.get("no_logprob", 0.0) or 0.0),
                "step1_paired_exact_prob": 0.0,
                "step1_signal_quality": signal_quality,
                "step1_signal_source": str(
                    signal_info.get("signal_source", "logprobs_unavailable") or "logprobs_unavailable"
                ),
                "step1_routing_score": routing_score,
                "step1_binary_entropy": 0.0,
                "latency_ms": latency_ms,
                "usage": usage,
                "parse_failed": False,
                "parse_failure_reason": "",
            }
        return {
            "answer": answer,
            "raw_label": raw_label,
            "step1_yes_logprob": float(signal_info.get("yes_logprob", 0.0) or 0.0),
            "step1_no_logprob": float(signal_info.get("no_logprob", 0.0) or 0.0),
            "step1_paired_exact_prob": float(signal_info.get("paired_exact_prob", 0.0) or 0.0),
            "step1_signal_quality": signal_quality,
            "step1_signal_source": str(
                signal_info.get("signal_source", "paired_y_n_top_logprobs") or "paired_y_n_top_logprobs"
            ),
            "step1_routing_score": routing_score,
            "step1_binary_entropy": float(signal_info.get("binary_entropy", 0.0) or 0.0),
            "latency_ms": latency_ms,
            "usage": usage,
            "parse_failed": False,
            "parse_failure_reason": "",
        }
    def forward_batch(
        self,
        samples: List[Sample],
        *,
        max_parallel_requests: int,
    ) -> List[Dict[str, Any]]:
        if not samples:
            return []
        total = len(samples)
        parallel = max(1, min(total, int(max_parallel_requests)))
        if self._vllm_client is None:
            self.load()
        if parallel <= 1 or total <= 1:
            return [
                self._forward_vllm_single_token(
                    str(getattr(sample, "prompt_evidence_text", "") or sample.text),
                    sample.query,
                    task_label=str(getattr(sample, "prompt_task_label", "Query") or "Query"),
                    evidence_label=str(getattr(sample, "prompt_evidence_label", "Document") or "Document"),
                    task_text=str(
                        getattr(sample, "prompt_task_text", "")
                        or getattr(sample, "claim_text", "")
                        or sample.query
                    ),
                    instruction_text=str(getattr(sample, "prompt_instruction_text", "") or ""),
                )
                for sample in samples[:total]
            ]
        def _infer_one(sample: Sample) -> Dict[str, Any]:
            return self._forward_vllm_single_token(
                str(getattr(sample, "prompt_evidence_text", "") or sample.text),
                sample.query,
                task_label=str(getattr(sample, "prompt_task_label", "Query") or "Query"),
                evidence_label=str(getattr(sample, "prompt_evidence_label", "Document") or "Document"),
                task_text=str(
                    getattr(sample, "prompt_task_text", "")
                    or getattr(sample, "claim_text", "")
                    or sample.query
                ),
                instruction_text=str(getattr(sample, "prompt_instruction_text", "") or ""),
            )
        return _parallel_map_ordered(
            items=list(samples[:total]),
            fn=_infer_one,
            max_workers=parallel,
        )

class Step1Judge:
    def __init__(self, config: Any, models: LocalModelManager):
        self.config = config
        self.models = models
        self._last_batch_usage: Dict[str, int] = {
            "llm_calls_total": 0,
            "prompt_tokens_total": 0,
            "system_prompt_tokens_total": 0,
            "user_prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
        }
    def get_last_batch_usage(self) -> Dict[str, int]:
        return dict(self._last_batch_usage)
    def judge_batch(self, samples: List[Sample]) -> List[Sample]:
        to_judge = [sample for sample in samples if sample.action != Action.ACCEPT]
        if not to_judge:
            self._last_batch_usage = {
                "llm_calls_total": 0,
                "prompt_tokens_total": 0,
                "system_prompt_tokens_total": 0,
                "user_prompt_tokens_total": 0,
                "completion_tokens_total": 0,
                "total_tokens_total": 0,
            }
            return samples
        results = self.models.forward_batch(
            to_judge,
            max_parallel_requests=int(self.config.step1_max_parallel_requests),
        )
        self._last_batch_usage = {
            "llm_calls_total": int(len(results)),
            "prompt_tokens_total": int(
                sum(
                    int(
                        (
                            result.get("usage", {})
                            if isinstance(result.get("usage"), dict)
                            else {}
                        ).get("prompt_tokens", 0)
                        or 0
                    )
                    for result in results
                    if isinstance(result, dict)
                )
            ),
            "system_prompt_tokens_total": int(
                sum(
                    int(
                        (
                            result.get("usage", {})
                            if isinstance(result.get("usage"), dict)
                            else {}
                        ).get("system_prompt_tokens", 0)
                        or 0
                    )
                    for result in results
                    if isinstance(result, dict)
                )
            ),
            "user_prompt_tokens_total": int(
                sum(
                    int(
                        (
                            result.get("usage", {})
                            if isinstance(result.get("usage"), dict)
                            else {}
                        ).get("user_prompt_tokens", 0)
                        or 0
                    )
                    for result in results
                    if isinstance(result, dict)
                )
            ),
            "completion_tokens_total": int(
                sum(
                    int(
                        (
                            result.get("usage", {})
                            if isinstance(result.get("usage"), dict)
                            else {}
                        ).get("completion_tokens", 0)
                        or 0
                    )
                    for result in results
                    if isinstance(result, dict)
                )
            ),
            "total_tokens_total": int(
                sum(
                    int(
                        (
                            result.get("usage", {})
                            if isinstance(result.get("usage"), dict)
                            else {}
                        ).get("total_tokens", 0)
                        or 0
                    )
                    for result in results
                    if isinstance(result, dict)
                )
            ),
        }
        for sample, result in zip(to_judge, results):
            apply_step1_result(sample, result)
        return samples

def infer_chunk_step1_batch(
    batch_payload: dict[str, Any],
    *,
    client: Any,
    model: str,
    score_temperature: float = DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE,
) -> dict[str, Any]:
    batch_id = str(batch_payload.get("batch_id", "") or "")
    items = [item for item in list(batch_payload.get("items", []) or []) if isinstance(item, dict)]
    if not items:
        return {"batch_id": batch_id, "items": []}
    prompt = create_chunk_step1_prompt(
        str(batch_payload.get("query_text", "") or ""),
        items,
        task_label=str(batch_payload.get("prompt_task_label", "Query") or "Query"),
        evidence_label=str(batch_payload.get("prompt_evidence_label", "Document") or "Document"),
        instruction_text=str(batch_payload.get("prompt_instruction_text", "") or ""),
    )
    system_prompt = build_chunk_step1_system_prompt(
        task_label=str(batch_payload.get("prompt_task_label", "Query") or "Query"),
        evidence_label=str(batch_payload.get("prompt_evidence_label", "Document") or "Document"),
        instruction_text=str(batch_payload.get("prompt_instruction_text", "") or ""),
    )
    max_output_tokens = 8 if len(items) == 1 else max(32, len(items) * 12)
    result = vllm_chat_result(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_output_tokens,
        disable_thinking=True,
        logprobs=True,
        top_logprobs=8,
    )
    usage = dict(result.get("usage", {}) or {})
    prompt_breakdown = estimate_prompt_token_breakdown(system_texts=[system_prompt], user_texts=[prompt])
    usage["system_prompt_tokens"] = int(prompt_breakdown["system_prompt_tokens"])
    usage["user_prompt_tokens"] = int(prompt_breakdown["user_prompt_tokens"])
    if not int(usage.get("prompt_tokens", 0) or 0):
        usage["prompt_tokens"] = int(prompt_breakdown["prompt_tokens"])
    labels_by_index = parse_chunk_step1_output(
        str(result.get("content", "") or ""),
        len(items),
    )
    expected_labels = [labels_by_index[index] for index in range(1, len(items) + 1)]
    score_payload = extract_chunk_step1_label_scores(
        result.get("raw_response"),
        expected_labels,
    )
    if not bool(score_payload.get("score_ok", False)):
        raise RuntimeError(str(score_payload.get("error", "") or "chunk_step1_score_error"))
    score_items = list(score_payload.get("items", []) or [])
    normalized_rows: list[dict[str, Any]] = []
    for display_index, item in enumerate(items, start=1):
        prediction = labels_by_index.get(display_index, "")
        normalized_rows.append(
            {
                "sample_id": str(item.get("sample_id", "") or ""),
                "prediction": prediction,
                "normalized": normalize_chunk_step1_score_row(
                    score_items[display_index - 1],
                    prediction,
                    score_temperature=float(score_temperature),
                    batch_item_count=len(items),
                ),
            }
        )
    return {
        "batch_id": batch_id,
        "items": normalized_rows,
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
    }

def _settings_from_args(args: Namespace) -> CascadeSettings:
    return CascadeSettings(
        command="step1",
        mode=str(args.mode),
        queries=str(args.queries),
        documents=str(args.documents),
        ground_truth=str(getattr(args, "ground_truth", "") or ""),
        artifacts_root=str(args.artifacts_root),
        embedding_model=str(args.embedding_model),
        step1_chat_model=str(args.step1_chat_model),
        step1_chat_base_url=str(args.step1_chat_base_url),
        step1_chat_local_path=str(getattr(args, "step1_chat_local_path", "") or ""),
        step1_max_parallel_requests=int(getattr(args, "step1_max_parallel_requests", 0) or 0),
        step1_chunk_score_temperature=float(
            getattr(args, "step1_chunk_score_temperature", DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE)
        ),
    )

def _chunk_batch_sample_ids(batch_payload: dict[str, Any]) -> set[str]:
    raw_ids = list(batch_payload.get("sample_ids", []) or [])
    if raw_ids:
        return {str(sample_id) for sample_id in raw_ids if str(sample_id or "")}
    return {
        str(item.get("sample_id", "") or "")
        for item in list(batch_payload.get("items", []) or [])
        if isinstance(item, dict) and str(item.get("sample_id", "") or "")
    }

def _get_chunk_step1_client(settings: CascadeSettings, config: _Step1Config) -> Any:
    local_path = str(settings.step1_chat_local_path or "").strip()
    if local_path:
        return _ensure_local_vllm_chat_ready(
            {
                "scope": "cascade_step1_chunk",
                "model": str(settings.step1_chat_model),
                "api_key": "EMPTY",
                "base_url": str(settings.step1_chat_base_url),
                "timeout": 120,
                "local_model_path": local_path,
                "max_num_seqs": int(config.step1_server_max_num_seqs),
                "gpu_memory_utilization": float(config.step1_server_gpu_memory_utilization),
            }
        )
    return get_openai_compatible_client(
        api_key="EMPTY",
        base_url=settings.step1_chat_base_url,
        timeout=120,
    )

def _empty_step1_usage() -> dict[str, Any]:
    return {
        "llm_calls_total": 0,
        "prompt_tokens_total": 0,
        "system_prompt_tokens_total": 0,
        "user_prompt_tokens_total": 0,
        "completion_tokens_total": 0,
        "total_tokens_total": 0,
        "batch_request_count": 0,
        "document_count": 0,
        "failed_document_count": 0,
        "token_accounting_mode": "provider_usage",
    }

def _add_step1_usage(total: dict[str, Any], batch_usage: dict[str, Any]) -> None:
    llm_calls = int(batch_usage.get("llm_calls_total", 0) or 0)
    for key in (
        "llm_calls_total",
        "prompt_tokens_total",
        "system_prompt_tokens_total",
        "user_prompt_tokens_total",
        "completion_tokens_total",
        "total_tokens_total",
    ):
        total[key] = int(total.get(key, 0) or 0) + int(batch_usage.get(key, 0) or 0)
    total["batch_request_count"] = int(total.get("batch_request_count", 0) or 0) + int(
        batch_usage.get("batch_request_count", llm_calls) or 0
    )
    total["document_count"] = int(total.get("document_count", 0) or 0) + int(
        batch_usage.get("document_count", llm_calls) or 0
    )
    total["failed_document_count"] = int(total.get("failed_document_count", 0) or 0) + int(
        batch_usage.get("failed_document_count", 0) or 0
    )

def _append_document_step1_path(sample: Sample) -> None:
    sample.path.append(
        f"step1:document(pred={sample.stage1_prediction},"
        f"score={float(sample.step1_routing_score or 0.0):.3f},"
        f"signal={sample.step1_signal_quality or 'unknown'})"
    )

def _mark_document_step1_exception(sample: Sample, exc: BaseException) -> None:
    reason = str(exc) or exc.__class__.__name__
    apply_step1_result(
        sample,
        {
            "answer": None,
            "raw_label": "",
            "step1_yes_logprob": 0.0,
            "step1_no_logprob": 0.0,
            "step1_paired_exact_prob": 0.0,
            "step1_signal_quality": "parse_failed",
            "step1_signal_source": "document_step1_exception",
            "step1_routing_score": 0.0,
            "step1_binary_entropy": 0.0,
            "latency_ms": 0.0,
            "usage": {},
            "parse_failed": True,
            "parse_failure_reason": reason,
        },
    )

def _run_document_step1(
    settings: CascadeSettings,
    samples: list[Sample],
    *,
    writer: StreamingRunWriter | None = None,
    completed_offset: int = 0,
    progress_total: int | None = None,
    resumed_count: int = 0,
) -> dict[str, Any]:
    config = _Step1Config.from_settings(settings)
    models = LocalModelManager(config)
    judge = Step1Judge(config, models)
    usage = _empty_step1_usage()
    flush_size = max(1, int(config.step1_max_parallel_requests or 1))
    total_for_progress = int(progress_total if progress_total is not None else len(samples))
    processed = 0
    flushed_batch_index = 0
    def _flush_batch(batch: list[Sample], batch_usage: dict[str, Any]) -> None:
        nonlocal processed, flushed_batch_index
        flushed_batch_index += 1
        _add_step1_usage(usage, batch_usage)
        for sample in batch:
            _append_document_step1_path(sample)
            if writer is not None:
                writer.write_stage_row("step1_rows", step1_stage_row(sample))
        processed += len(batch)
        if writer is not None:
            completed = int(completed_offset + processed)
            writer.write_progress(
                stage="step1",
                completed=completed,
                total=total_for_progress,
                mode=settings.mode,
                resumed=int(resumed_count),
            )
            writer.write_event(
                "step1_document_batch_finished",
                mode=settings.mode,
                batch_index=flushed_batch_index,
                batch_size=len(batch),
                completed=completed,
                total=total_for_progress,
                resumed=int(resumed_count),
            )
    def _run_batch(batch: list[Sample]) -> None:
        if not batch:
            return
        try:
            judge.judge_batch(batch)
        except Exception as exc:
            if len(batch) <= 1:
                _mark_document_step1_exception(batch[0], exc)
                _flush_batch(batch, {"failed_document_count": 1})
                return
            midpoint = max(1, len(batch) // 2)
            if writer is not None:
                writer.write_event(
                    "step1_document_batch_split_after_error",
                    mode=settings.mode,
                    batch_size=len(batch),
                    left_size=midpoint,
                    right_size=len(batch) - midpoint,
                    error_type=exc.__class__.__name__,
                    error=str(exc)[:500],
                    completed=int(completed_offset + processed),
                    total=total_for_progress,
                    resumed=int(resumed_count),
                )
            _run_batch(batch[:midpoint])
            _run_batch(batch[midpoint:])
            return
        _flush_batch(batch, judge.get_last_batch_usage())
    for start in range(0, len(samples), flush_size):
        _run_batch(samples[start : start + flush_size])
    return usage

def _run_chunk_step1(settings: CascadeSettings, samples: list[Sample]) -> dict[str, Any]:
    pending_ids = {sample.sample_id for sample in samples}
    by_sample_id = {sample.sample_id: sample for sample in samples}
    prepared_batch_payloads = [
        batch
        for batch in read_jsonl(Path(settings.artifacts_root) / "step1_batch_inputs.jsonl")
        if _chunk_batch_sample_ids(batch) & pending_ids
    ]
    batch_token_budget = _resolve_chunk_step1_batch_token_budget()
    batch_payloads = _rebatch_chunk_step1_payloads(
        prepared_batch_payloads,
        pending_ids=pending_ids,
        token_budget=batch_token_budget,
    )
    if not batch_payloads:
        return {
            "llm_calls_total": 0,
            "prompt_tokens_total": 0,
            "system_prompt_tokens_total": 0,
            "user_prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "batch_request_count": 0,
            "document_count": 0,
            "token_accounting_mode": "provider_usage",
        }
    config = _Step1Config.from_settings(settings)
    client = _get_chunk_step1_client(settings, config)
    def _infer(batch_payload: dict[str, Any]) -> dict[str, Any]:
        return infer_chunk_step1_batch(
            batch_payload,
            client=client,
            model=str(settings.step1_chat_model),
            score_temperature=float(settings.step1_chunk_score_temperature),
        )
    parallel = int(config.step1_max_parallel_requests)
    if parallel > 1 and len(batch_payloads) > 1:
        batch_outputs = _parallel_map_ordered(
            items=batch_payloads,
            fn=_infer,
            max_workers=parallel,
        )
    else:
        batch_outputs = [_infer(batch_payload) for batch_payload in batch_payloads]
    usage = {
        "llm_calls_total": int(len(batch_outputs)),
        "prompt_tokens_total": int(
            sum(int((dict(output.get("usage", {}) or {})).get("prompt_tokens", 0) or 0) for output in batch_outputs)
        ),
        "system_prompt_tokens_total": int(
            sum(
                int((dict(output.get("usage", {}) or {})).get("system_prompt_tokens", 0) or 0)
                for output in batch_outputs
            )
        ),
        "user_prompt_tokens_total": int(
            sum(
                int((dict(output.get("usage", {}) or {})).get("user_prompt_tokens", 0) or 0)
                for output in batch_outputs
            )
        ),
        "completion_tokens_total": int(
            sum(int((dict(output.get("usage", {}) or {})).get("completion_tokens", 0) or 0) for output in batch_outputs)
        ),
        "total_tokens_total": int(
            sum(int((dict(output.get("usage", {}) or {})).get("total_tokens", 0) or 0) for output in batch_outputs)
        ),
        "batch_request_count": int(len(batch_outputs)),
        "document_count": int(sum(len(list(output.get("items", []) or [])) for output in batch_outputs)),
        "token_accounting_mode": "provider_usage",
    }
    for batch_output in batch_outputs:
        batch_id = str(batch_output.get("batch_id", "") or "")
        for item in list(batch_output.get("items", []) or []):
            if not isinstance(item, dict):
                continue
            sample = by_sample_id.get(str(item.get("sample_id", "") or ""))
            if sample is None:
                continue
            prediction = str(item.get("prediction", "") or "")
            normalized = dict(item.get("normalized", {}) or {})
            apply_step1_result(
                sample,
                {
                    "answer": prediction,
                    "raw_label": prediction,
                    "parse_failed": False,
                    "parse_failure_reason": "",
                    "step1_yes_logprob": normalized["yes_logprob"],
                    "step1_no_logprob": normalized["no_logprob"],
                    "step1_paired_exact_prob": normalized["paired_exact_prob"],
                    "step1_signal_quality": normalized["signal_quality"],
                    "step1_signal_source": (
                        "chunk_paired_y_n_top_logprobs"
                        if normalized["signal_quality"] == "retrieval_chunk"
                        else "chunk_single_sided_lower_bound"
                    ),
                    "step1_routing_score": normalized["routing_score"],
                    "step1_binary_entropy": normalized["binary_entropy"],
                },
            )
            sample.path.append(
                f"step1:chunk(batch_id={batch_id},pred={prediction},"
                f"score={sample.step1_routing_score:.3f},signal={sample.step1_signal_quality})"
            )
    return usage

def run_from_args(args: Namespace) -> dict[str, Any]:
    started_at = monotonic_seconds()
    settings = _settings_from_args(args)
    samples = load_prepared_samples(settings)
    output_dir = Path(settings.artifacts_root) / "stage_output"
    sample_ids = {sample.sample_id for sample in samples}
    filter_stage_rows_to_samples(output_dir, sample_ids)
    resumed_final_count = hydrate_final_rows(samples, output_dir)
    writer = StreamingRunWriter(output_dir)
    writer.write_event("step1_stage_started", mode=settings.mode, resumed_final_count=resumed_final_count)
    candidates = [sample for sample in samples if sample.action != Action.ACCEPT]
    resumed_ids = hydrate_step1_from_rows(candidates, writer.root)
    pending = [sample for sample in candidates if sample.sample_id not in resumed_ids]
    writer.write_event(
        "step1_started",
        mode=settings.mode,
        sample_count=len(pending),
        resumed_count=len(resumed_ids),
    )
    if pending:
        if settings.mode == "document":
            step1_usage = _run_document_step1(
                settings,
                pending,
                writer=writer,
                completed_offset=len(resumed_ids),
                progress_total=len(candidates),
                resumed_count=len(resumed_ids),
            )
        else:
            step1_usage = _run_chunk_step1(settings, pending)
            for sample in pending:
                writer.write_stage_row("step1_rows", step1_stage_row(sample))
    else:
        step1_usage = _empty_step1_usage()
    writer.write_progress(
        stage="step1",
        completed=len(candidates),
        total=len(candidates),
        mode=settings.mode,
        resumed=len(resumed_ids),
    )
    elapsed_seconds = elapsed_seconds_since(started_at)
    model_call_count = int(step1_usage.get("llm_calls_total", 0) or 0)
    cost = build_cost_summary(
        elapsed_seconds=elapsed_seconds,
        model_call_count=model_call_count,
        model_request_count=int(step1_usage.get("batch_request_count", model_call_count) or 0),
        model_input_count=int(step1_usage.get("document_count", len(pending) if model_call_count else 0) or 0),
        prompt_tokens=int(step1_usage.get("prompt_tokens_total", 0) or 0),
        system_prompt_tokens=int(step1_usage.get("system_prompt_tokens_total", 0) or 0),
        user_prompt_tokens=int(step1_usage.get("user_prompt_tokens_total", 0) or 0),
        completion_tokens=int(step1_usage.get("completion_tokens_total", 0) or 0),
    )
    payload = {
        "stage": "step1",
        "mode": str(settings.mode),
        "total_samples": int(len(samples)),
        "cached_final_count": int(resumed_final_count),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "step1_usage": step1_usage,
    }
    writer.write_summary(payload)
    write_json(output_dir / f"step1_{settings.mode}.json", payload)
    return payload
__all__ = [
    "infer_chunk_step1_batch",
    "LocalModelManager",
    "run_from_args",
    "Step1Judge",
]

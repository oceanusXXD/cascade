from __future__ import annotations
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from algorithm.core.step3 import parse_step3_batch_output
from algorithm.settings import CascadeSettings
from algorithm.common.artifacts import (
    filter_stage_rows_to_samples,
    hydrate_final_rows,
    hydrate_step1_from_rows,
    hydrate_step2_from_rows,
    latest_rows_by_sample_id,
    load_prepared_samples,
)
from algorithm.common.metrics import sample_to_result_row
from algorithm.common.runtime import (
    Action,
    Sample,
    Source,
    build_step3_batch_system_prompt,
)
from script.task_schema import resolve_prompt_payload
from script.data import load_ground_truth_index
from script.ids import split_sample_id
from script.io import write_json
from script.model_usage import (
    build_cost_summary,
    elapsed_seconds_since,
    estimate_prompt_token_breakdown,
    monotonic_seconds,
)
from script.stream import StreamingRunWriter
from script.vllm import get_openai_compatible_client, vllm_chat_result

def _step3_batch_query_ref(sample: Sample) -> Tuple[str, str]:
    sample_id = str(sample.sample_id or "").strip()
    if ":" in sample_id:
        _, query_id = split_sample_id(sample_id)
        return str(query_id), str(sample.query or "")
    return "", str(sample.query or "")

class CascadeStep3:
    MAX_TOKENS = 128
    def __init__(self, settings: CascadeSettings) -> None:
        self.settings = settings
        self.model = str(settings.step3_chat_model)
        self.base_url = str(settings.step3_chat_base_url)
        self.backend = "api"
        self.uses_chunk_json_protocol = bool(settings.mode == "chunk")
        self.max_tokens = int(self.MAX_TOKENS)
        self._client = None
        self._last_batch_usage: Dict[str, Any] = {
            "llm_calls_total": 0,
            "prompt_tokens_total": 0,
            "system_prompt_tokens_total": 0,
            "user_prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "token_accounting_mode": "provider_usage",
            "batch_mode": "single_document",
            "batch_request_count": 0,
            "document_count": 0,
            "per_batch_rows": [],
        }
        self._ground_truth_labels = (
            load_ground_truth_index(settings.ground_truth)
            if str(settings.ground_truth or "").strip()
            else {}
        )
        if self.model and self.base_url:
            self.backend = "api"
        elif self.model or self.base_url:
            raise ValueError("step3 requires step3_chat_model and step3_chat_base_url together")
        elif self._ground_truth_labels:
            self.backend = "ground_truth"
        else:
            raise ValueError("step3 requires step3_chat_model")
    def get_last_batch_usage(self) -> Dict[str, Any]:
        return dict(self._last_batch_usage)
    def _get_client(self) -> Any:
        if self._client is None:
            self._client = get_openai_compatible_client(
                api_key="EMPTY",
                base_url=self.base_url,
                timeout=120,
            )
        return self._client
    def _build_system_prompt(self, query: str) -> str:
        return build_step3_batch_system_prompt()
    def _build_batch_system_prompt(
        self,
        query: str,
        *,
        task_label: str = "Query",
        evidence_label: str = "Document",
        instruction_text: str = "",
        document_count: int = 0,
    ) -> str:
        base = self._build_system_prompt(query)
        sections = [base]
        instruction = str(instruction_text or "").strip()
        if instruction:
            sections.append(f"Additional task instruction:\n{instruction}")
        sections.append(f"Task field label: {str(task_label or 'Query')}.")
        sections.append(f"Evidence field label: {str(evidence_label or 'Document')}.")
        if self.uses_chunk_json_protocol:
            max_index = max(0, int(document_count) - 1)
            sections.append(
                "Evaluate each indexed evidence item independently. "
                "Mark an index as YES only when that evidence item itself contains direct evidence for YES. "
                "If an evidence item is ambiguous, weakly suggestive, or unrelated, treat it as NO. "
                f"yes_indices may only contain integers from 0 to {max_index}."
            )
            sections.append(
                'Output override: return valid JSON only in one of two formats: {"result":"all_no"} '
                'or {"result":"has_yes","yes_indices":[0,2]}. Do not return explanations or extra text.'
            )
            return "\n\n".join(sections)
        sections.append(
            "Output override: return exactly one line per indexed document in order, "
            "using only the format '<index>:Y' or '<index>:N'. "
            "Do not return JSON, explanations, or any extra text."
        )
        return "\n\n".join(sections)
    def _build_batch_prompt(self, samples: List[Sample]) -> str:
        if not samples:
            return ""
        payloads = [
            resolve_prompt_payload(
                sample,
                default_task_text=str(sample.query or ""),
                default_evidence_text=str(sample.text or ""),
            )
            for sample in samples
        ]
        first_payload = payloads[0]
        task_label = str(first_payload["task_label"] or "Query")
        evidence_label = str(first_payload["evidence_label"] or "Document")
        shared_task_text = str(first_payload["task_text"] or "")
        all_task_texts_match = all(
            str(payload["task_text"] or "") == shared_task_text
            for payload in payloads
        )
        if self.uses_chunk_json_protocol:
            lines = []
            if all_task_texts_match and shared_task_text:
                lines.append(f"{task_label}:\n{shared_task_text}")
            for local_idx, (sample, payload) in enumerate(zip(samples, payloads)):
                lines.append(f"[{local_idx}]")
                if not all_task_texts_match:
                    lines.append(f"{task_label}:")
                    lines.append(str(payload["task_text"] or ""))
                lines.append(f"{evidence_label}:")
                lines.append(str(payload["evidence_text"] or sample.text or ""))
            return "\n".join(lines)
        lines = []
        if all_task_texts_match and shared_task_text:
            lines.append(f"{task_label}:\n{shared_task_text}")
            lines.append("")
            for idx, (sample, payload) in enumerate(zip(samples, payloads), start=1):
                lines.append(f"[{idx}]")
                lines.append(str(payload["evidence_text"] or sample.text or ""))
                lines.append("")
        else:
            for idx, (sample, payload) in enumerate(zip(samples, payloads), start=1):
                lines.append(f"[{idx}]")
                lines.append(f"{task_label}:")
                lines.append(str(payload["task_text"] or ""))
                lines.append(f"{evidence_label}:")
                lines.append(str(payload["evidence_text"] or sample.text or ""))
                lines.append("")
        return "\n".join(lines).strip()
    def _parse_batch_output(self, text: str, document_count: int) -> List[str]:
        return parse_step3_batch_output(
            text,
            document_count,
            uses_chunk_json_protocol=bool(self.uses_chunk_json_protocol),
        )
    def _group_deferred_batches(self, deferred: List[Sample]) -> List[List[Sample]]:
        grouped: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
        for sample in deferred:
            grouped[_step3_batch_query_ref(sample)].append(sample)
        return list(grouped.values())
    def _infer_batch(self, batch: List[Sample]) -> Tuple[List[str], Dict[str, int]]:
        first_payload = resolve_prompt_payload(
            batch[0],
            default_task_text=str(batch[0].query or ""),
            default_evidence_text=str(batch[0].text or ""),
        )
        system_query = str(
            first_payload["task_text"]
            or first_payload["instruction_text"]
            or batch[0].query
            or ""
        )
        system_prompt = self._build_batch_system_prompt(
            system_query,
            task_label=str(first_payload["task_label"] or "Query"),
            evidence_label=str(first_payload["evidence_label"] or "Document"),
            instruction_text=str(first_payload["instruction_text"] or ""),
            document_count=len(batch),
        )
        user_prompt = self._build_batch_prompt(batch)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = vllm_chat_result(
            client=self._get_client(),
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
            disable_thinking=True,
        )
        labels = self._parse_batch_output(str(result.get("content", "")), len(batch))
        usage = dict(result.get("usage", {}) or {})
        prompt_breakdown = estimate_prompt_token_breakdown(system_texts=[system_prompt], user_texts=[user_prompt])
        usage["system_prompt_tokens"] = int(prompt_breakdown["system_prompt_tokens"])
        usage["user_prompt_tokens"] = int(prompt_breakdown["user_prompt_tokens"])
        if not int(usage.get("prompt_tokens", 0) or 0):
            usage["prompt_tokens"] = int(prompt_breakdown["prompt_tokens"])
        return labels, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "system_prompt_tokens": int(usage.get("system_prompt_tokens", 0) or 0),
            "user_prompt_tokens": int(usage.get("user_prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    def process_deferred(
        self,
        samples: List[Sample],
        on_sample_done: Callable[[Sample], None] | None = None,
    ) -> List[Sample]:
        deferred = [sample for sample in samples if sample.action == Action.DEFER]
        if self.backend == "ground_truth":
            self._last_batch_usage = {
                "llm_calls_total": 0,
                "prompt_tokens_total": 0,
                "system_prompt_tokens_total": 0,
                "user_prompt_tokens_total": 0,
                "completion_tokens_total": 0,
                "total_tokens_total": 0,
                "token_accounting_mode": "provider_usage",
                "batch_mode": "ground_truth_labels",
                "batch_request_count": 0,
                "document_count": int(len(deferred)),
                "per_batch_rows": [],
            }
            if not deferred:
                return samples
            missing = [
                sample.sample_id
                for sample in deferred
                if self._ground_truth_labels.get(sample.sample_id, "") not in {"yes", "no"}
            ]
            if missing:
                raise RuntimeError(
                    "ground truth labels missing for deferred samples: " + ", ".join(sorted(missing))
                )
            for sample in deferred:
                answer = str(self._ground_truth_labels.get(sample.sample_id, "") or "")
                sample.action = Action.ACCEPT
                sample.prediction = answer
                sample.source = Source.STEP3_GROUND_TRUTH
                if sample.defer_reason:
                    sample.path.append(f"step3:from_defer(reason={sample.defer_reason})")
                sample.path.append(f"step3:ground_truth({answer})")
                if on_sample_done is not None:
                    on_sample_done(sample)
            return samples
        self._last_batch_usage = {
            "llm_calls_total": 0,
            "prompt_tokens_total": 0,
            "system_prompt_tokens_total": 0,
            "user_prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "token_accounting_mode": "provider_usage",
            "batch_mode": (
                "document_level_all_no_has_yes" if self.uses_chunk_json_protocol else "document_level_indexed_yn"
            ),
            "batch_request_count": 0,
            "document_count": int(len(deferred)),
            "per_batch_rows": [],
        }
        if not deferred:
            return samples
        batches = self._group_deferred_batches(deferred)
        self._last_batch_usage["batch_request_count"] = int(len(batches))
        for batch_idx, batch in enumerate(batches):
            outputs, usage = self._infer_batch(batch)
            self._last_batch_usage["llm_calls_total"] += 1
            self._last_batch_usage["prompt_tokens_total"] += int(usage.get("prompt_tokens", 0) or 0)
            self._last_batch_usage["system_prompt_tokens_total"] += int(usage.get("system_prompt_tokens", 0) or 0)
            self._last_batch_usage["user_prompt_tokens_total"] += int(usage.get("user_prompt_tokens", 0) or 0)
            self._last_batch_usage["completion_tokens_total"] += int(usage.get("completion_tokens", 0) or 0)
            self._last_batch_usage["total_tokens_total"] += int(usage.get("total_tokens", 0) or 0)
            self._last_batch_usage["per_batch_rows"].append(
                {
                    "batch_idx": int(batch_idx),
                    "document_count": int(len(batch)),
                    "query_id": str(_step3_batch_query_ref(batch[0])[0] if batch else ""),
                    "query": str(batch[0].query or "") if batch else "",
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "system_prompt_tokens": int(usage.get("system_prompt_tokens", 0) or 0),
                    "user_prompt_tokens": int(usage.get("user_prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                }
            )
            for sample, answer in zip(batch, outputs):
                sample.action = Action.ACCEPT
                sample.prediction = answer
                sample.source = Source.STEP3_LARGE_MODEL
                if sample.defer_reason:
                    sample.path.append(f"step3:from_defer(reason={sample.defer_reason})")
                sample.path.append(f"step3:model({answer})")
                if on_sample_done is not None:
                    on_sample_done(sample)
        return samples

def _settings_from_args(args: Namespace) -> CascadeSettings:
    return CascadeSettings(
        command="step3",
        mode=str(args.mode),
        queries=str(args.queries),
        documents=str(args.documents),
        ground_truth=str(getattr(args, "ground_truth", "") or ""),
        artifacts_root=str(args.artifacts_root),
        embedding_model=str(args.embedding_model),
        step3_chat_model=str(getattr(args, "step3_chat_model", "") or ""),
        step3_chat_base_url=str(getattr(args, "step3_chat_base_url", "") or ""),
    )

def run_from_args(args: Namespace) -> dict[str, Any]:
    started_at = monotonic_seconds()
    settings = _settings_from_args(args)
    samples = load_prepared_samples(settings)
    output_dir = Path(settings.artifacts_root) / "stage_output"
    sample_ids = {sample.sample_id for sample in samples}
    filter_stage_rows_to_samples(output_dir, sample_ids)
    resumed_final_count = hydrate_final_rows(samples, output_dir)
    writer = StreamingRunWriter(output_dir)
    writer.write_event("step3_stage_started", mode=settings.mode, resumed_final_count=resumed_final_count)
    candidates = [sample for sample in samples if sample.action != Action.ACCEPT]
    hydrate_step1_from_rows(candidates, writer.root)
    resumed_step2_ids = hydrate_step2_from_rows(candidates, writer.root)
    missing_step2_ids = sorted(
        sample.sample_id
        for sample in candidates
        if sample.sample_id not in resumed_step2_ids
    )
    if missing_step2_ids:
        raise RuntimeError(
            "step3 requires cached step2 rows for samples: "
            + ", ".join(missing_step2_ids[:10])
        )
    emitted_final_ids = set(
        latest_rows_by_sample_id(
            writer.root,
            "final_rows.jsonl",
            {sample.sample_id for sample in samples},
        )
    )
    def _write_final_once(sample: Sample) -> None:
        sample_id = str(sample.sample_id or "")
        if sample_id in emitted_final_ids:
            return
        writer.write_stage_row("final_rows", sample_to_result_row(sample))
        emitted_final_ids.add(sample_id)
    for sample in samples:
        if sample.action == Action.ACCEPT:
            _write_final_once(sample)
    runner = CascadeStep3(settings)
    outputs = runner.process_deferred(samples, on_sample_done=_write_final_once)
    for sample in outputs:
        if sample.action == Action.ACCEPT:
            _write_final_once(sample)
    writer.write_progress(stage="complete", completed=len(outputs), total=len(outputs), mode=settings.mode)
    step3_usage = runner.get_last_batch_usage()
    elapsed_seconds = elapsed_seconds_since(started_at)
    model_call_count = int(step3_usage.get("llm_calls_total", 0) or 0)
    cost = build_cost_summary(
        elapsed_seconds=elapsed_seconds,
        model_call_count=model_call_count,
        model_request_count=int(step3_usage.get("batch_request_count", model_call_count) or 0),
        model_input_count=int(step3_usage.get("document_count", 0) or 0) if model_call_count else 0,
        prompt_tokens=int(step3_usage.get("prompt_tokens_total", 0) or 0),
        system_prompt_tokens=int(step3_usage.get("system_prompt_tokens_total", 0) or 0),
        user_prompt_tokens=int(step3_usage.get("user_prompt_tokens_total", 0) or 0),
        completion_tokens=int(step3_usage.get("completion_tokens_total", 0) or 0),
    )
    payload = {
        "stage": "step3",
        "mode": str(settings.mode),
        "total_samples": int(len(outputs)),
        "cached_final_count": int(resumed_final_count),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
        "step3_usage": step3_usage,
        "results": [sample_to_result_row(sample) for sample in sorted(outputs, key=lambda sample: sample.sample_id)],
    }
    writer.write_summary(payload)
    write_json(output_dir / f"step3_{settings.mode}.json", payload)
    return payload
__all__ = ["CascadeStep3", "run_from_args"]

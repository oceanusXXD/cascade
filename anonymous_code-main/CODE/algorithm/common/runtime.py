from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
STEP1_ROUTABLE_SIGNAL_QUALITIES = {
    "paired_exact",
    "single_sided_lb",
    "retrieval_chunk",
}
STEP1_SINGLE_TOKEN_SYSTEM_PROMPT = "You are a binary classifier. Output only Y or N. Y means yes. N means no."
STEP3_BATCH_SYSTEM_PROMPT = """
You are a strict document analysis assistant for yes/no classification.
The user message contains only query/document data to judge.
Base every decision only on explicit evidence in the provided documents.
Never rely on unstated assumptions or world knowledge.
When the evidence is missing, partial, indirect, or ambiguous, prefer NO.
""".strip()

class Action(str, Enum):
    ACCEPT = "accept"
    CONTINUE = "continue"
    DEFER = "defer"

class Source(str, Enum):
    STEP1_LOCAL = "step1_local"
    STEP2_LOCAL = "step2_local"
    STEP3_LARGE_MODEL = "step3_large_model"
    STEP3_GROUND_TRUTH = "step3_ground_truth"

@dataclass

class Sample:
    sample_id: str
    text: str
    query: str
    label: Optional[str] = None
    raw_query_text: str = ""
    prompt_style: str = "query_document"
    prompt_task_label: str = "Query"
    prompt_evidence_label: str = "Document"
    claim_text: str = ""
    prompt_task_text: str = ""
    prompt_instruction_text: str = ""
    prompt_evidence_text: str = ""
    embedding_task_text: str = ""
    embedding_instruction_text: str = ""
    embedding_evidence_text: str = ""
    prediction: Optional[str] = None
    action: Action = Action.CONTINUE
    source: Optional[Source] = None
    stage1_prediction: Optional[str] = None
    stage2_prediction: Optional[str] = None
    stage2_routing_score: float = 0.0
    step1_raw_label: str = ""
    step1_parse_failed: bool = False
    step1_parse_failure_reason: str = ""
    step1_yes_logprob: float = 0.0
    step1_no_logprob: float = 0.0
    step1_paired_exact_prob: float = 0.0
    step1_signal_quality: str = ""
    step1_signal_source: str = ""
    step1_neighbor_support: float = 0.0
    step1_neighbor_k_eff: float = 0.0
    step1_neighbor_bar_w: float = 0.0
    step1_neighbor_rho: float = 0.0
    step1_tilde_b: float = 0.0
    step2_neighbor_support: float = 0.0
    step2_neighbor_k_eff: float = 0.0
    step2_neighbor_bar_w: float = 0.0
    step2_neighbor_rho: float = 0.0
    step2_tilde_b: float = 0.0
    step1_routing_score: float = 0.0
    step1_binary_entropy: float = 0.0
    defer_reason: str = ""
    step2_decision_threshold_used: Optional[float] = None
    step2_lambda_hat_used: Optional[float] = None
    path: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    @property
    def document(self) -> str:
        return str(self.text or "")
    @document.setter
    def document(self, value: Any) -> None:
        self.text = str(value or "")
    @property
    def document_text(self) -> str:
        return str(self.text or "")
    @document_text.setter
    def document_text(self, value: Any) -> None:
        self.text = str(value or "")
    @property
    def query_text(self) -> str:
        return str(self.query or "")
    @query_text.setter
    def query_text(self, value: Any) -> None:
        self.query = str(value or "")

def apply_step1_result(
    sample: Sample,
    result: Dict[str, Any],
) -> None:
    answer = str(result.get("answer", "") or "").strip().lower()
    signal_quality = str(result.get("step1_signal_quality", "") or "")
    routing_score = float(result.get("step1_routing_score", 0.0) or 0.0)
    sample.stage1_prediction = answer if answer in {"yes", "no"} else None
    sample.step1_raw_label = str(result.get("raw_label", "") or "").strip()
    sample.step1_parse_failed = bool(result.get("parse_failed", False))
    sample.step1_parse_failure_reason = str(result.get("parse_failure_reason", "") or "")
    sample.step1_yes_logprob = float(result.get("step1_yes_logprob", 0.0) or 0.0)
    sample.step1_no_logprob = float(result.get("step1_no_logprob", 0.0) or 0.0)
    sample.step1_paired_exact_prob = float(result.get("step1_paired_exact_prob", 0.0) or 0.0)
    sample.step1_signal_quality = signal_quality
    sample.step1_signal_source = str(result.get("step1_signal_source", "") or "")
    sample.step1_neighbor_support = float(result.get("step1_neighbor_support", 0.0) or 0.0)
    sample.step1_neighbor_k_eff = float(result.get("step1_neighbor_k_eff", 0.0) or 0.0)
    sample.step1_neighbor_bar_w = float(result.get("step1_neighbor_bar_w", 0.0) or 0.0)
    sample.step1_neighbor_rho = float(result.get("step1_neighbor_rho", 0.0) or 0.0)
    sample.step1_tilde_b = float(result.get("step1_tilde_b", 0.0) or 0.0)
    sample.step2_neighbor_support = float(result.get("step2_neighbor_support", 0.0) or 0.0)
    sample.step2_neighbor_k_eff = float(result.get("step2_neighbor_k_eff", 0.0) or 0.0)
    sample.step2_neighbor_bar_w = float(result.get("step2_neighbor_bar_w", 0.0) or 0.0)
    sample.step2_neighbor_rho = float(result.get("step2_neighbor_rho", 0.0) or 0.0)
    sample.step2_tilde_b = float(result.get("step2_tilde_b", 0.0) or 0.0)
    sample.step1_routing_score = routing_score
    sample.step1_binary_entropy = float(result.get("step1_binary_entropy", 0.0) or 0.0)
    sample.latency_ms += float(result.get("latency_ms", 0.0) or 0.0)

def build_step1_system_prompt() -> str:
    return STEP1_SINGLE_TOKEN_SYSTEM_PROMPT

def build_step3_batch_system_prompt() -> str:
    return STEP3_BATCH_SYSTEM_PROMPT

def build_step1_prompt(
    text: str,
    query: str,
    *,
    task_label: str = "Query",
    evidence_label: str = "Document",
    task_text: str = "",
    instruction_text: str = "",
) -> str:
    effective_task_text = str(task_text or query or "").strip()
    return f"Query: {effective_task_text}\nDocument:\n{str(text or '').strip()}\nAnswer:"
__all__ = [
    "Action",
    "apply_step1_result",
    "STEP1_ROUTABLE_SIGNAL_QUALITIES",
    "STEP1_SINGLE_TOKEN_SYSTEM_PROMPT",
    "STEP3_BATCH_SYSTEM_PROMPT",
    "Sample",
    "Source",
    "build_step1_prompt",
    "build_step1_system_prompt",
    "build_step3_batch_system_prompt",
]

from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
DEFAULT_STEP1_BATCH_SIZE = 10
DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE = 10.0
DEFAULT_STEP1_BATCH_PROTOCOL_VERSION = "chunk_step1_v1"
_ALLOWED_MODES = {"document", "chunk"}
_ALLOWED_COMMANDS = {"prepare", "step1", "step2", "step3"}

@dataclass(frozen=True)

class CascadeSettings:
    mode: str
    queries: str
    documents: str
    artifacts_root: str
    command: str = "prepare"
    step1_chat_model: str = ""
    step1_chat_base_url: str = ""
    step2_chat_model: str = ""
    step2_chat_base_url: str = ""
    step2_chat_local_path: str = ""
    step2_max_parallel_requests: int = 0
    step3_chat_model: str = ""
    step3_chat_base_url: str = ""
    embedding_model: str = ""
    embedding_base_url: str = ""
    tau: float = 0.95
    delta_w: float = 0.05
    combiner_source: str = "source1"
    combiner_theta_grid: str = ""
    step1_batch_size: int = DEFAULT_STEP1_BATCH_SIZE
    step1_chunk_score_temperature: float = DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE
    step1_max_parallel_requests: int = 0
    ground_truth: str = ""
    step1_chat_local_path: str = ""
    def __post_init__(self) -> None:
        command = str(self.command or "").strip().lower() or "prepare"
        if command not in _ALLOWED_COMMANDS:
            raise ValueError(f"unsupported algorithm command: {self.command!r}")
        mode = str(self.mode or "").strip().lower()
        if mode not in _ALLOWED_MODES:
            raise ValueError(f"unsupported algorithm mode: {self.mode!r}")
        step1_chat_model = str(self.step1_chat_model or "").strip()
        step1_chat_base_url = str(self.step1_chat_base_url or "").strip()
        if command == "step1":
            if not step1_chat_model:
                raise ValueError("step1_chat_model is required for step1")
            if not step1_chat_base_url:
                raise ValueError("step1_chat_base_url is required for step1")
        step2_chat_model = str(self.step2_chat_model or "").strip()
        step2_chat_base_url = str(self.step2_chat_base_url or "").strip()
        if command == "step2" and bool(step2_chat_model) != bool(step2_chat_base_url):
            raise ValueError("step2_chat_model and step2_chat_base_url must be provided together for step2")
        step3_chat_model = str(self.step3_chat_model or "").strip()
        step3_chat_base_url = str(self.step3_chat_base_url or "").strip()
        ground_truth = str(self.ground_truth or "").strip()
        if command == "step3":
            has_step3_model = bool(step3_chat_model)
            has_step3_base_url = bool(step3_chat_base_url)
            if has_step3_model != has_step3_base_url:
                raise ValueError("step3_chat_model and step3_chat_base_url must be provided together for step3")
            if not has_step3_model and not ground_truth:
                raise ValueError("step3_chat_model is required for step3 unless ground_truth is provided")
        embedding_model = str(self.embedding_model or "").strip()
        if not embedding_model:
            raise ValueError("embedding_model is required")
        embedding_base_url = str(self.embedding_base_url or "").strip()
        if command == "prepare" and not embedding_base_url and not Path(embedding_model).exists():
            raise ValueError(
                "embedding_base_url is required unless embedding_model is a local model path"
            )
        resolved_step1_batch_size = int(self.step1_batch_size or DEFAULT_STEP1_BATCH_SIZE)
        resolved_tau = float(self.tau)
        if not math.isfinite(resolved_tau) or not (0.0 <= resolved_tau <= 1.0):
            raise ValueError("tau must be within [0, 1]")
        resolved_delta_w = float(self.delta_w)
        if not math.isfinite(resolved_delta_w) or not (
            0.0 < resolved_delta_w <= 1.0
        ):
            raise ValueError("delta_w must be within (0, 1]")
        combiner_source = str(self.combiner_source or "source1").strip().lower()
        if combiner_source in {"source2", "ii", "2", "theta", "theta_grid", "grid"}:
            combiner_source = "source2"
        else:
            combiner_source = "source1"
        resolved_step1_chunk_score_temperature = float(
            getattr(self, "step1_chunk_score_temperature", DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE)
        )
        if not math.isfinite(resolved_step1_chunk_score_temperature):
            raise ValueError("step1_chunk_score_temperature must be finite")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "step1_chat_model", step1_chat_model)
        object.__setattr__(self, "step1_chat_base_url", step1_chat_base_url)
        object.__setattr__(self, "step2_chat_model", step2_chat_model)
        object.__setattr__(self, "step2_chat_base_url", step2_chat_base_url)
        object.__setattr__(self, "step2_chat_local_path", str(self.step2_chat_local_path or "").strip())
        object.__setattr__(self, "step2_max_parallel_requests", max(0, int(self.step2_max_parallel_requests or 0)))
        object.__setattr__(self, "step3_chat_model", step3_chat_model)
        object.__setattr__(self, "step3_chat_base_url", step3_chat_base_url)
        object.__setattr__(self, "embedding_model", embedding_model)
        object.__setattr__(self, "embedding_base_url", embedding_base_url)
        object.__setattr__(self, "tau", float(resolved_tau))
        object.__setattr__(self, "delta_w", float(resolved_delta_w))
        object.__setattr__(self, "combiner_source", combiner_source)
        object.__setattr__(self, "combiner_theta_grid", str(self.combiner_theta_grid or "").strip())
        object.__setattr__(self, "artifacts_root", str(Path(self.artifacts_root)))
        object.__setattr__(self, "ground_truth", ground_truth)
        object.__setattr__(self, "step1_chat_local_path", str(self.step1_chat_local_path or "").strip())
        object.__setattr__(self, "step1_batch_size", max(1, int(resolved_step1_batch_size)))
        object.__setattr__(self, "step1_chunk_score_temperature", float(resolved_step1_chunk_score_temperature))
        object.__setattr__(self, "step1_max_parallel_requests", max(0, int(self.step1_max_parallel_requests or 0)))

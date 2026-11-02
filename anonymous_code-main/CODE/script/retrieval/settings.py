from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from script.vllm import get_local_vllm_gpu_profile
DEFAULT_RERANKER_MODEL = ""
DEFAULT_RERANK_INSTRUCTION = "Given a query, judge whether the document is relevant to the query."
DEFAULT_CHUNK_SELECTOR_VERSION = "rerank_sentence_window_v1"

@dataclass(frozen=True)

class ChunkSelectionSettings:
    queries: str
    documents: str
    artifacts_root: str
    ground_truth: str = ""
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = "EMPTY"
    embedding_timeout: int = 180
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_base_url: str = ""
    reranker_api_key: str = "EMPTY"
    reranker_timeout: int = 180
    reranker_instruction: str = DEFAULT_RERANK_INSTRUCTION
    reranker_max_length: int = 8192
    rerank_request_batch_size: int = 0
    window_size: int = 3
    rrf_k: int = 60
    use_hyde_cache: bool = False
    use_query_rewrite_cache: bool = False
    def __post_init__(self) -> None:
        embedding_model = str(self.embedding_model or "").strip()
        embedding_base_url = str(self.embedding_base_url or "").strip()
        if embedding_model and not embedding_base_url and not Path(embedding_model).exists():
            raise ValueError("embedding_base_url is required unless embedding_model is a local model path")
        embedding_timeout = 180 if self.embedding_timeout is None else int(self.embedding_timeout)
        reranker_model = str(self.reranker_model or DEFAULT_RERANKER_MODEL).strip()
        reranker_timeout = 180 if self.reranker_timeout is None else int(self.reranker_timeout)
        reranker_max_length = 8192 if self.reranker_max_length is None else int(self.reranker_max_length)
        gpu_profile = get_local_vllm_gpu_profile() or {}
        profile_rerank_request_batch_size = max(
            1,
            int(gpu_profile.get("rerank_request_batch_size", 16) or 16),
        )
        rerank_request_batch_size = (
            profile_rerank_request_batch_size
            if self.rerank_request_batch_size is None or int(self.rerank_request_batch_size) <= 0
            else int(self.rerank_request_batch_size)
        )
        window_size = 3 if self.window_size is None else int(self.window_size)
        rrf_k = 60 if self.rrf_k is None else int(self.rrf_k)
        object.__setattr__(self, "artifacts_root", str(Path(self.artifacts_root)))
        object.__setattr__(self, "embedding_model", embedding_model)
        object.__setattr__(self, "embedding_base_url", embedding_base_url)
        object.__setattr__(self, "embedding_api_key", str(self.embedding_api_key or "EMPTY").strip() or "EMPTY")
        object.__setattr__(self, "reranker_model", reranker_model)
        object.__setattr__(self, "reranker_base_url", str(self.reranker_base_url or "").strip())
        object.__setattr__(self, "reranker_api_key", str(self.reranker_api_key or "EMPTY").strip() or "EMPTY")
        object.__setattr__(self, "reranker_timeout", max(1, reranker_timeout))
        object.__setattr__(
            self,
            "reranker_instruction",
            str(self.reranker_instruction or DEFAULT_RERANK_INSTRUCTION).strip(),
        )
        object.__setattr__(self, "reranker_max_length", max(128, reranker_max_length))
        object.__setattr__(self, "rerank_request_batch_size", max(1, rerank_request_batch_size))
        object.__setattr__(self, "embedding_timeout", max(1, embedding_timeout))
        object.__setattr__(self, "window_size", max(1, window_size))
        object.__setattr__(self, "rrf_k", max(1, rrf_k))
        object.__setattr__(self, "use_hyde_cache", bool(self.use_hyde_cache))
        object.__setattr__(self, "use_query_rewrite_cache", bool(self.use_query_rewrite_cache))
__all__ = [
    "ChunkSelectionSettings",
    "DEFAULT_CHUNK_SELECTOR_VERSION",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RERANK_INSTRUCTION",
]

from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Any, Sequence
import gc
import numpy as np
from .vllm_common import get_local_vllm_gpu_profile, get_openai_compatible_client
_LOCAL_VLLM_EMBEDDING_RUNNER_CACHE: dict[tuple[str, bool, float, int, int, int, bool], Any] = {}
_LOCAL_VLLM_EMBEDDING_RUNNER_LOCK = threading.Lock()

def _prepare_local_vllm_embedding_allocator() -> None:
    gpu_profile = get_local_vllm_gpu_profile() or {}
    if not bool(gpu_profile.get("embed_cuda_expandable_segments", False)):
        return
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _local_vllm_embedding_pooler_config(*, normalize: bool = True) -> Any:
    from vllm.config import PoolerConfig
    return PoolerConfig(
        seq_pooling_type="LAST",
        tok_pooling_type="ALL",
        use_activation=bool(normalize),
    )

def _local_vllm_embedding_settings() -> tuple[float, int, int, int, bool]:
    def _env_float(name: str, default: float) -> float:
        try:
            return float(str(os.environ.get(name, default)).strip())
        except (TypeError, ValueError):
            return float(default)
    def _env_int(name: str, default: int) -> int:
        try:
            return int(str(os.environ.get(name, default)).strip())
        except (TypeError, ValueError):
            return int(default)
    def _env_bool(name: str, default: bool) -> bool:
        raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
        return raw not in {"", "0", "false", "no", "off"}
    def _env_has_value(name: str) -> bool:
        return str(os.environ.get(name, "") or "").strip() != ""
    gpu_profile = get_local_vllm_gpu_profile() or {}
    default_gpu_memory = float(gpu_profile.get("gpu_memory_utilization", 0.70) or 0.70)
    default_max_num_seqs = max(1, int(gpu_profile.get("embed_max_num_seqs", 8) or 8))
    default_max_num_batched_tokens = max(0, int(gpu_profile.get("embed_max_num_batched_tokens", 0) or 0))
    default_max_model_len = max(0, int(gpu_profile.get("embed_max_model_len", 0) or 0))
    default_enforce_eager = bool(gpu_profile.get("embed_enforce_eager", True))
    if _env_has_value("CASCADE_LOCAL_VLLM_EMBED_GPU_MEMORY_UTILIZATION"):
        gpu_memory_utilization = _env_float("CASCADE_LOCAL_VLLM_EMBED_GPU_MEMORY_UTILIZATION", default_gpu_memory)
    else:
        gpu_memory_utilization = default_gpu_memory
    gpu_memory_utilization = max(0.05, min(float(gpu_memory_utilization), 0.98))
    if _env_has_value("CASCADE_LOCAL_VLLM_EMBED_MAX_NUM_SEQS"):
        max_num_seqs = _env_int("CASCADE_LOCAL_VLLM_EMBED_MAX_NUM_SEQS", default_max_num_seqs)
    else:
        max_num_seqs = default_max_num_seqs
    max_num_seqs = max(1, int(max_num_seqs))
    if _env_has_value("CASCADE_LOCAL_VLLM_EMBED_MAX_NUM_BATCHED_TOKENS"):
        max_num_batched_tokens = _env_int(
            "CASCADE_LOCAL_VLLM_EMBED_MAX_NUM_BATCHED_TOKENS",
            default_max_num_batched_tokens,
        )
    else:
        max_num_batched_tokens = default_max_num_batched_tokens
    max_num_batched_tokens = max(0, int(max_num_batched_tokens))
    if _env_has_value("CASCADE_LOCAL_VLLM_EMBED_MAX_MODEL_LEN"):
        max_model_len = _env_int("CASCADE_LOCAL_VLLM_EMBED_MAX_MODEL_LEN", default_max_model_len)
    else:
        max_model_len = default_max_model_len
    max_model_len = max(0, int(max_model_len))
    if _env_has_value("CASCADE_LOCAL_VLLM_EMBED_ENFORCE_EAGER"):
        enforce_eager = _env_bool("CASCADE_LOCAL_VLLM_EMBED_ENFORCE_EAGER", default_enforce_eager)
    else:
        enforce_eager = default_enforce_eager
    return (
        float(gpu_memory_utilization),
        int(max_num_seqs),
        int(max_num_batched_tokens),
        int(max_model_len),
        bool(enforce_eager),
    )

def _local_vllm_embedding_tokenization_kwargs() -> dict[str, int | bool] | None:
    raw = str(os.environ.get("CASCADE_LOCAL_VLLM_EMBED_TRUNCATE_TOKENS", "") or "").strip()
    if not raw:
        return None
    try:
        max_length = int(raw)
    except (TypeError, ValueError):
        max_length = 0
    if max_length <= 0:
        return None
    return {"truncation": True, "max_length": int(max_length)}

def _should_use_local_vllm_embedding_backend(model: str) -> bool:
    return Path(str(model or "").strip()).exists()

def _get_local_vllm_embedding_runner(model_path: str, *, normalize: bool = True) -> Any:
    _prepare_local_vllm_embedding_allocator()
    from vllm import LLM
    normalized_model_path = str(Path(model_path).resolve())
    (
        gpu_memory_utilization,
        max_num_seqs,
        max_num_batched_tokens,
        max_model_len,
        enforce_eager,
    ) = _local_vllm_embedding_settings()
    cache_key = (
        normalized_model_path,
        bool(normalize),
        float(gpu_memory_utilization),
        int(max_num_seqs),
        int(max_num_batched_tokens),
        int(max_model_len),
        bool(enforce_eager),
    )
    with _LOCAL_VLLM_EMBEDDING_RUNNER_LOCK:
        runner = _LOCAL_VLLM_EMBEDDING_RUNNER_CACHE.get(cache_key)
        if runner is not None:
            return runner
        llm_kwargs = {
            "model": normalized_model_path,
            "runner": "pooling",
            "convert": "embed",
            "pooler_config": _local_vllm_embedding_pooler_config(normalize=normalize),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "max_num_seqs": int(max_num_seqs),
            "enforce_eager": bool(enforce_eager),
        }
        if int(max_num_batched_tokens) > 0:
            llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if int(max_model_len) > 0:
            llm_kwargs["max_model_len"] = int(max_model_len)
        runner = LLM(**llm_kwargs)
        _LOCAL_VLLM_EMBEDDING_RUNNER_CACHE[cache_key] = runner
        return runner

def release_local_vllm_embedding_runner_cache() -> None:
    with _LOCAL_VLLM_EMBEDDING_RUNNER_LOCK:
        runners = list(_LOCAL_VLLM_EMBEDDING_RUNNER_CACHE.values())
        _LOCAL_VLLM_EMBEDDING_RUNNER_CACHE.clear()
    for runner in runners:
        engine = getattr(runner, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
    del runners
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

def _embed_texts_with_local_vllm(
    *,
    model_path: str,
    texts: Sequence[str],
    request_batch_size: int,
    normalize: bool,
) -> np.ndarray:
    runner = _get_local_vllm_embedding_runner(model_path, normalize=normalize)
    vectors: list[list[float]] = []
    effective_batch = max(1, int(request_batch_size))
    items = [str(text or "") for text in texts]
    tokenization_kwargs = _local_vllm_embedding_tokenization_kwargs()
    for start in range(0, len(items), effective_batch):
        batch_texts = items[start : start + effective_batch]
        embed_kwargs: dict[str, Any] = {"use_tqdm": False}
        if tokenization_kwargs is not None:
            embed_kwargs["tokenization_kwargs"] = tokenization_kwargs
        try:
            outputs = list(runner.embed(batch_texts, **embed_kwargs))
        except TypeError as exc:
            if "use_tqdm" not in str(exc):
                raise
            embed_kwargs.pop("use_tqdm", None)
            outputs = list(runner.embed(batch_texts, **embed_kwargs))
        batch_vectors = [
            list(getattr(getattr(item, "outputs", None), "embedding", []) or [])
            for item in outputs
        ]
        if len(batch_vectors) != len(batch_texts):
            raise RuntimeError(
                f"local embedding response size mismatch: expected={len(batch_texts)} actual={len(batch_vectors)}"
            )
        vectors.extend(batch_vectors)
    arr = np.asarray(vectors, dtype=np.float32)
    if normalize and arr.size > 0:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        arr = arr / norms
    return arr.astype(np.float32)

def vllm_embed_texts(
    *,
    base_url: str,
    model: str,
    texts: Sequence[str],
    api_key: str = "EMPTY",
    timeout: int = 180,
    request_batch_size: int = 32,
    batch_size: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    items = [str(text or "") for text in texts]
    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    effective_request_batch_size = int(
        batch_size if batch_size is not None else request_batch_size
    )
    if _should_use_local_vllm_embedding_backend(model):
        return _embed_texts_with_local_vllm(
            model_path=str(model),
            texts=items,
            request_batch_size=effective_request_batch_size,
            normalize=bool(normalize),
        )
    client = get_openai_compatible_client(
        api_key=str(api_key or "EMPTY"),
        base_url=str(base_url),
        timeout=int(timeout),
    )
    vectors: list[list[float]] = []
    effective_batch = max(1, int(effective_request_batch_size))
    for start in range(0, len(items), effective_batch):
        batch_texts = items[start : start + effective_batch]
        response = client.embeddings.create(
            model=str(model),
            input=batch_texts,
        )
        batch_vectors = [list(item.embedding) for item in response.data]
        if len(batch_vectors) != len(batch_texts):
            raise RuntimeError(
                f"embedding response size mismatch: expected={len(batch_texts)} actual={len(batch_vectors)}"
            )
        vectors.extend(batch_vectors)
    arr = np.asarray(vectors, dtype=np.float32)
    if normalize and arr.size > 0:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        arr = arr / norms
    return arr.astype(np.float32)
__all__ = ["release_local_vllm_embedding_runner_cache", "vllm_embed_texts"]

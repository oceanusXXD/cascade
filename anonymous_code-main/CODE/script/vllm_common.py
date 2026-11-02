from __future__ import annotations
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Sequence
from urllib.parse import urlparse
_OPENAI_CLIENT_CACHE: Dict[str, Any] = {}
_OPENAI_CLIENT_LOCK = threading.Lock()
_LOCAL_VLLM_GPU_PROFILE_CACHE: Dict[str, Dict[str, Any] | None] = {}
_LOCAL_VLLM_GPU_PROFILE_LOCK = threading.Lock()

def get_openai_compatible_client(
    *,
    api_key: str,
    base_url: str,
    timeout: int,
) -> Any:
    from openai import OpenAI
    resolved_api_key = str(api_key or "EMPTY")
    resolved_base_url = str(base_url)
    resolved_timeout = int(timeout)
    cache_key = f"{resolved_base_url}|{resolved_api_key}|{resolved_timeout}"
    with _OPENAI_CLIENT_LOCK:
        client = _OPENAI_CLIENT_CACHE.get(cache_key)
        if client is not None:
            return client
        client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=resolved_timeout,
        )
        _OPENAI_CLIENT_CACHE[cache_key] = client
        return client

def create_chat_completion(
    *,
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    disable_thinking: bool = True,
    **kwargs: Any,
) -> Any:
    request_params: Dict[str, Any] = {
        "model": str(model),
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        **kwargs,
    }
    normalized_model = str(model).strip().lower()
    if disable_thinking and normalized_model.startswith("qwen"):
        extra_body = dict(request_params.get("extra_body") or {})
        extra_body.setdefault("enable_thinking", False)
        base_url_text = str(getattr(client, "base_url", "") or "").strip()
        host = urlparse(base_url_text).hostname or ""
        if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_template_kwargs.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        request_params["extra_body"] = extra_body
    return client.chat.completions.create(**request_params)

def extract_chat_content(response: Any) -> str:
    return str(response.choices[0].message.content or "")

def extract_response_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }

def parallel_map_ordered(
    *,
    items: Sequence[Any],
    fn: Callable[[Any], Any],
    max_workers: int,
) -> List[Any]:
    if not items:
        return []
    workers = max(1, min(int(max_workers), len(items)))
    if workers <= 1:
        return [fn(item) for item in items]
    results: List[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
    return results

def _normalize_gpu_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())

def _speed_profile(
    *,
    gpu_family: str,
    gpu_memory_utilization: float,
    chat_max_model_len: int,
    chat_max_num_seqs: int,
    chat_max_num_batched_tokens: int,
    chat_parallel_requests: int,
    embed_max_model_len: int,
    embed_max_num_seqs: int,
    embed_max_num_batched_tokens: int,
    embed_request_batch_size: int,
    embed_inflight_batches: int,
    embed_cuda_expandable_segments: bool,
    rerank_max_model_len: int,
    rerank_max_num_seqs: int,
    rerank_max_num_batched_tokens: int,
    rerank_request_batch_size: int,
) -> Dict[str, Any]:
    return {
        "gpu_family": str(gpu_family),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "chat_max_model_len": int(chat_max_model_len),
        "chat_max_num_seqs": int(chat_max_num_seqs),
        "chat_max_num_batched_tokens": int(chat_max_num_batched_tokens),
        "chat_parallel_requests": int(chat_parallel_requests),
        "chat_enforce_eager": True,
        "embed_max_model_len": int(embed_max_model_len),
        "embed_max_num_seqs": int(embed_max_num_seqs),
        "embed_max_num_batched_tokens": int(embed_max_num_batched_tokens),
        "embed_request_batch_size": int(embed_request_batch_size),
        "embed_inflight_batches": int(embed_inflight_batches),
        "embed_cuda_expandable_segments": bool(embed_cuda_expandable_segments),
        "embed_enforce_eager": True,
        "rerank_max_model_len": int(rerank_max_model_len),
        "rerank_max_num_seqs": int(rerank_max_num_seqs),
        "rerank_max_num_batched_tokens": int(rerank_max_num_batched_tokens),
        "rerank_request_batch_size": int(rerank_request_batch_size),
        "rerank_enforce_eager": True,
    }

def resolve_local_vllm_gpu_profile(gpu_name: str, gpu_memory_mb: int | None = None) -> Dict[str, Any] | None:
    normalized = _normalize_gpu_name(gpu_name)
    if not normalized:
        return None
    if "l4" in normalized:
        return _speed_profile(
            gpu_family="l4",
            gpu_memory_utilization=0.95,
            chat_max_model_len=4096,
            chat_max_num_seqs=64,
            chat_max_num_batched_tokens=16384,
            chat_parallel_requests=8,
            embed_max_model_len=4096,
            embed_max_num_seqs=512,
            embed_max_num_batched_tokens=32768,
            embed_request_batch_size=256,
            embed_inflight_batches=2,
            embed_cuda_expandable_segments=False,
            rerank_max_model_len=4096,
            rerank_max_num_seqs=256,
            rerank_max_num_batched_tokens=32768,
            rerank_request_batch_size=64,
        )
    is_rtx_6000_class = (
        "rtx pro 6000" in normalized
        or "rtx p6000" in normalized
        or ("rtx" in normalized and "6000" in normalized and int(gpu_memory_mb or 0) >= 90000)
    )
    if is_rtx_6000_class:
        return _speed_profile(
            gpu_family="rtx_p6000_max",
            gpu_memory_utilization=0.95,
            chat_max_model_len=4096,
            chat_max_num_seqs=512,
            chat_max_num_batched_tokens=65536,
            chat_parallel_requests=32,
            embed_max_model_len=4096,
            embed_max_num_seqs=4096,
            embed_max_num_batched_tokens=327680,
            embed_request_batch_size=1024,
            embed_inflight_batches=8,
            embed_cuda_expandable_segments=True,
            rerank_max_model_len=4096,
            rerank_max_num_seqs=1024,
            rerank_max_num_batched_tokens=131072,
            rerank_request_batch_size=256,
        )
    return None

def _query_local_vllm_gpu_name() -> tuple[str, int] | tuple[str, None]:
    override = str(os.environ.get("CASCADE_LOCAL_VLLM_GPU_NAME", "") or "").strip()
    if override:
        return override, None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "", None
    if completed.returncode != 0:
        return "", None
    first_line = ""
    for raw_line in str(completed.stdout or "").splitlines():
        probe = str(raw_line or "").strip()
        if probe:
            first_line = probe
            break
    if not first_line:
        return "", None
    if "," not in first_line:
        return first_line.strip(), None
    name_text, memory_text = first_line.split(",", 1)
    try:
        memory_mb = int(float(str(memory_text or "").strip()))
    except (TypeError, ValueError):
        memory_mb = None
    return name_text.strip(), memory_mb

def get_local_vllm_gpu_profile() -> Dict[str, Any] | None:
    detected_name, detected_memory_mb = _query_local_vllm_gpu_name()
    cache_key = f"{detected_name}|{detected_memory_mb}"
    with _LOCAL_VLLM_GPU_PROFILE_LOCK:
        cached = _LOCAL_VLLM_GPU_PROFILE_CACHE.get(cache_key, None)
        if cache_key in _LOCAL_VLLM_GPU_PROFILE_CACHE:
            return dict(cached) if isinstance(cached, dict) else None
        profile = resolve_local_vllm_gpu_profile(detected_name, detected_memory_mb)
        if profile is not None:
            profile = {
                **profile,
                "detected_gpu_name": str(detected_name or ""),
                "detected_gpu_memory_mb": detected_memory_mb,
            }
        _LOCAL_VLLM_GPU_PROFILE_CACHE[cache_key] = dict(profile) if isinstance(profile, dict) else None
        return dict(profile) if isinstance(profile, dict) else None
__all__ = [
    "create_chat_completion",
    "extract_chat_content",
    "extract_response_usage",
    "get_local_vllm_gpu_profile",
    "get_openai_compatible_client",
    "parallel_map_ordered",
    "resolve_local_vllm_gpu_profile",
]

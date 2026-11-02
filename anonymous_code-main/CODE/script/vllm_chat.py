from __future__ import annotations
import importlib
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from .vllm_common import (
    create_chat_completion,
    extract_chat_content,
    extract_response_usage,
    get_local_vllm_gpu_profile,
    get_openai_compatible_client,
)
logger = logging.getLogger(__name__)
_LOCAL_VLLM_PROCESS_CACHE: Dict[str, subprocess.Popen[str]] = {}
_LOCAL_VLLM_LOG_HANDLE_CACHE: Dict[str, Any] = {}
_LOCAL_VLLM_TARGET_MODEL_CACHE: Dict[str, str] = {}
_LOCAL_VLLM_LOCK = threading.Lock()

def vllm_chat_result(
    *,
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    disable_thinking: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    response = create_chat_completion(
        client=client,
        model=str(model),
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        disable_thinking=bool(disable_thinking),
        **kwargs,
    )
    return {
        "content": extract_chat_content(response),
        "usage": extract_response_usage(response),
        "raw_response": response,
    }

def query_served_model_ids(client: Any) -> set[str]:
    models_resp = client.models.list()
    return {
        str(item.id)
        for item in getattr(models_resp, "data", [])
        if getattr(item, "id", None)
    }

def _ensure_flash_attn_available() -> bool:
    try:
        importlib.import_module("flash_attn")
        return True
    except Exception as exc:
        logger.warning(
            "flash_attn is not available in the active environment; "
            "continuing local runtime startup without preflight enforcement: %s",
            exc,
        )
        return False

def _require_psutil() -> Any:
    try:
        import psutil
        return psutil
    except Exception as exc:
        raise RuntimeError("psutil is required for local runtime auto-switch and process control") from exc

def _is_local_base_url(base_url: str) -> bool:
    raw = str(base_url or "").strip()
    if raw and "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    host = str(parsed.hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

def _resolve_host_port(base_url: str) -> Tuple[str, int]:
    raw = str(base_url or "").strip()
    if raw and "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    return str(parsed.hostname or "127.0.0.1"), int(parsed.port or 8000)

def _default_log_file(base_url: str, scope: str, served_model_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "output" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(served_model_name))
    _, port = _resolve_host_port(base_url)
    return logs_dir / f"local_runtime_chat_{scope}_{port}_{safe_model}.log"

def _preferred_local_vllm_python() -> str:
    override = str(os.environ.get("CASCADE_LOCAL_VLLM_PYTHON", "") or "").strip()
    if override:
        return override
    repo_python = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    if repo_python.exists():
        return str(repo_python)
    return str(sys.executable)

def _resolve_gpu_memory_utilization(raw_value: Any, *, default: float = 0.90) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float(default)
    if value <= 0.0:
        return float(default)
    if value < float(default):
        return float(default)
    return float(value)

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

def _build_local_vllm_command(
    *,
    model_path: str,
    host: str,
    port: int,
    served_model_name: str,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    python_executable: str,
    max_model_len: int = 0,
    max_num_batched_tokens: int = 0,
    enforce_eager: bool = False,
    trust_remote_code: bool = False,
) -> list[str]:
    cmd = [
        str(python_executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--host",
        str(host),
        "--port",
        str(port),
        "--served-model-name",
        str(served_model_name),
        "--gpu-memory-utilization",
        f"{max(0.05, min(float(gpu_memory_utilization), 0.98)):.3f}",
        "--max-num-seqs",
        str(max(1, int(max_num_seqs))),
    ]
    if int(max_model_len) > 0:
        cmd.extend(["--max-model-len", str(int(max_model_len))])
    if int(max_num_batched_tokens) > 0:
        cmd.extend(["--max-num-batched-tokens", str(int(max_num_batched_tokens))])
    if bool(enforce_eager):
        cmd.append("--enforce-eager")
    if bool(trust_remote_code):
        cmd.append("--trust-remote-code")
    return cmd

def _normalize_chat_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    base_url = str(settings.get("base_url", "") or "").strip()
    model = str(settings.get("model", "") or "").strip()
    if not base_url:
        raise ValueError("local runtime settings.base_url is required")
    if not model:
        raise ValueError("local runtime settings.model is required")
    scope = str(settings.get("scope", "algorithm") or "algorithm").strip() or "algorithm"
    log_file = str(settings.get("log_file", "") or "").strip()
    if not log_file:
        log_file = str(_default_log_file(base_url, scope, model))
    gpu_profile = get_local_vllm_gpu_profile() or {}
    profile_gpu_memory = float(gpu_profile.get("gpu_memory_utilization", 0.90) or 0.90)
    profile_max_num_seqs = max(1, int(gpu_profile.get("chat_max_num_seqs", 16) or 16))
    profile_max_model_len = max(0, int(gpu_profile.get("chat_max_model_len", 0) or 0))
    profile_max_num_batched_tokens = max(0, int(gpu_profile.get("chat_max_num_batched_tokens", 0) or 0))
    profile_enforce_eager = bool(gpu_profile.get("chat_enforce_eager", False))
    if _env_has_value("CASCADE_LOCAL_VLLM_CHAT_MAX_NUM_SEQS"):
        requested_max_num_seqs = max(1, _env_int("CASCADE_LOCAL_VLLM_CHAT_MAX_NUM_SEQS", profile_max_num_seqs))
    else:
        requested_max_num_seqs = max(1, int(settings.get("max_num_seqs", profile_max_num_seqs) or profile_max_num_seqs))
    if gpu_profile and not _env_has_value("CASCADE_LOCAL_VLLM_CHAT_MAX_NUM_SEQS"):
        requested_max_num_seqs = min(requested_max_num_seqs, profile_max_num_seqs)
    if _env_has_value("CASCADE_LOCAL_VLLM_CHAT_MAX_MODEL_LEN"):
        resolved_max_model_len = max(0, _env_int("CASCADE_LOCAL_VLLM_CHAT_MAX_MODEL_LEN", 0))
    else:
        requested_max_model_len = max(0, int(settings.get("max_model_len", profile_max_model_len) or 0))
        if gpu_profile and profile_max_model_len > 0:
            resolved_max_model_len = (
                profile_max_model_len
                if requested_max_model_len <= 0
                else min(requested_max_model_len, profile_max_model_len)
            )
        else:
            resolved_max_model_len = requested_max_model_len
    if _env_has_value("CASCADE_LOCAL_VLLM_CHAT_MAX_NUM_BATCHED_TOKENS"):
        resolved_max_num_batched_tokens = max(
            0,
            _env_int("CASCADE_LOCAL_VLLM_CHAT_MAX_NUM_BATCHED_TOKENS", 0),
        )
    else:
        requested_max_num_batched_tokens = max(
            0,
            int(settings.get("max_num_batched_tokens", profile_max_num_batched_tokens) or 0),
        )
        if gpu_profile and profile_max_num_batched_tokens > 0:
            resolved_max_num_batched_tokens = (
                profile_max_num_batched_tokens
                if requested_max_num_batched_tokens <= 0
                else min(requested_max_num_batched_tokens, profile_max_num_batched_tokens)
            )
        else:
            resolved_max_num_batched_tokens = requested_max_num_batched_tokens
    if _env_has_value("CASCADE_LOCAL_VLLM_CHAT_ENFORCE_EAGER"):
        resolved_enforce_eager = _env_bool("CASCADE_LOCAL_VLLM_CHAT_ENFORCE_EAGER", False)
    else:
        resolved_enforce_eager = bool(settings.get("enforce_eager", profile_enforce_eager))
        if gpu_profile:
            resolved_enforce_eager = bool(resolved_enforce_eager or profile_enforce_eager)
    if _env_has_value("CASCADE_LOCAL_VLLM_CHAT_TRUST_REMOTE_CODE"):
        resolved_trust_remote_code = _env_bool("CASCADE_LOCAL_VLLM_CHAT_TRUST_REMOTE_CODE", False)
    else:
        resolved_trust_remote_code = bool(settings.get("trust_remote_code", False))
    return {
        "scope": scope,
        "model": model,
        "api_key": str(settings.get("api_key", "EMPTY") or "EMPTY"),
        "base_url": base_url,
        "timeout": max(1, int(settings.get("timeout", 120) or 120)),
        "local_model_path": str(settings.get("local_model_path", "") or "").strip(),
        "log_file": log_file,
        "max_num_seqs": requested_max_num_seqs,
        "max_model_len": resolved_max_model_len,
        "max_num_batched_tokens": resolved_max_num_batched_tokens,
        "enforce_eager": resolved_enforce_eager,
        "trust_remote_code": resolved_trust_remote_code,
        "gpu_memory_utilization": _resolve_gpu_memory_utilization(
            settings.get("gpu_memory_utilization", profile_gpu_memory),
            default=profile_gpu_memory,
        ),
    }

def _list_listener_pids(port: int) -> list[int]:
    psutil = _require_psutil()
    pids: set[int] = set()
    for conn in psutil.net_connections(kind="tcp"):
        if conn.status != psutil.CONN_LISTEN:
            continue
        laddr = conn.laddr
        conn_port = getattr(laddr, "port", None)
        if conn_port is None and isinstance(laddr, tuple) and len(laddr) >= 2:
            conn_port = laddr[1]
        if int(conn_port or 0) != int(port):
            continue
        if conn.pid and int(conn.pid) != os.getpid():
            pids.add(int(conn.pid))
    return sorted(pids)

def _stop_process_tree(pid: int, timeout_s: float = 20.0) -> None:
    psutil = _require_psutil()
    try:
        root = psutil.Process(int(pid))
    except psutil.Error:
        return
    children = root.children(recursive=True)
    for proc in reversed(children):
        try:
            proc.terminate()
        except psutil.Error:
            pass
    try:
        root.terminate()
    except psutil.Error:
        pass
    _, alive = psutil.wait_procs(children + [root], timeout=max(1.0, float(timeout_s)))
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5.0)

def _stop_port_owners(port: int) -> None:
    for pid in _list_listener_pids(port):
        _stop_process_tree(pid)

def _close_cached_handles(base_url: str) -> None:
    handle = _LOCAL_VLLM_LOG_HANDLE_CACHE.pop(base_url, None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            logger.debug("failed to close local runtime log handle", exc_info=True)

def _start_local_vllm_chat_service(settings: Dict[str, Any]) -> subprocess.Popen[str]:
    base_url = str(settings["base_url"])
    if not _is_local_base_url(base_url):
        raise RuntimeError(
            f"local runtime auto-switch only supports localhost base_url, got: {base_url}"
        )
    _ensure_flash_attn_available()
    model_path = str(settings["local_model_path"] or "").strip()
    if not model_path:
        raise RuntimeError(
            f"models.{settings.get('scope', 'unknown')}.local_path is required for local runtime auto-switch"
        )
    host, port = _resolve_host_port(base_url)
    cmd = _build_local_vllm_command(
        model_path=model_path,
        host=host,
        port=port,
        served_model_name=str(settings["model"]),
        gpu_memory_utilization=float(settings["gpu_memory_utilization"]),
        max_num_seqs=int(settings["max_num_seqs"]),
        python_executable=_preferred_local_vllm_python(),
        max_model_len=int(settings.get("max_model_len", 0) or 0),
        max_num_batched_tokens=int(settings.get("max_num_batched_tokens", 0) or 0),
        enforce_eager=bool(settings.get("enforce_eager", False)),
        trust_remote_code=bool(settings.get("trust_remote_code", False)),
    )
    log_path = Path(str(settings["log_file"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8")
    launch_env = os.environ.copy()
    launch_env.setdefault("PYTHONNOUSERSITE", "1")
    launch_env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=launch_env,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        close_fds=True,
        text=True,
        start_new_session=True,
    )
    _LOCAL_VLLM_LOG_HANDLE_CACHE[base_url] = log_handle
    logger.warning(
        "local runtime chat service launched: pid=%s base_url=%s model=%s path=%s python=%s log=%s",
        str(proc.pid),
        base_url,
        str(settings["model"]),
        model_path,
        str(cmd[0]),
        str(log_path),
    )
    return proc

def _restart_local_vllm_chat_service(settings: Dict[str, Any]) -> subprocess.Popen[str]:
    base_url = str(settings["base_url"])
    _, port = _resolve_host_port(base_url)
    cached_proc = _LOCAL_VLLM_PROCESS_CACHE.pop(base_url, None)
    if cached_proc is not None and cached_proc.poll() is None:
        _stop_process_tree(int(cached_proc.pid))
    _stop_port_owners(port)
    _close_cached_handles(base_url)
    proc = _start_local_vllm_chat_service(settings)
    _LOCAL_VLLM_PROCESS_CACHE[base_url] = proc
    _LOCAL_VLLM_TARGET_MODEL_CACHE[base_url] = str(settings["model"])
    return proc

def ensure_local_vllm_chat_ready(settings: Dict[str, Any]) -> Any:
    resolved = _normalize_chat_settings(settings)
    client = get_openai_compatible_client(
        api_key=str(resolved["api_key"]),
        base_url=str(resolved["base_url"]),
        timeout=int(resolved["timeout"]),
    )
    base_url = str(resolved["base_url"])
    expected_model = str(resolved["model"]).strip()
    cached_target = str(_LOCAL_VLLM_TARGET_MODEL_CACHE.get(base_url, "") or "").strip()
    if cached_target and cached_target == expected_model:
        return client
    try:
        served_models = query_served_model_ids(client)
        if expected_model in served_models:
            logger.info(
                "local runtime chat ready at %s serving model=%s",
                base_url,
                expected_model,
            )
            _LOCAL_VLLM_TARGET_MODEL_CACHE[base_url] = expected_model
            return client
        if served_models:
            logger.warning(
                "local runtime at %s serves %s, expected %s; restarting for configured model",
                base_url,
                sorted(served_models),
                expected_model,
            )
    except Exception as exc:
        logger.warning("local runtime probe failed at %s, will try restart: %s", base_url, exc)
    if not _is_local_base_url(base_url):
        raise RuntimeError(
            f"local runtime at {base_url} is unavailable or serving the wrong model; "
            "automatic restart only supports localhost endpoints"
        )
    with _LOCAL_VLLM_LOCK:
        proc = _restart_local_vllm_chat_service(resolved)
    deadline = time.time() + float(max(int(resolved["timeout"]), 30))
    last_exc: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"failed to start local runtime chat process (exit={proc.returncode}) at {base_url}"
            ) from last_exc
        try:
            served_models = query_served_model_ids(client)
            if expected_model in served_models:
                logger.info(
                    "local runtime chat warmed up at %s serving model=%s",
                    base_url,
                    expected_model,
                )
                _LOCAL_VLLM_TARGET_MODEL_CACHE[base_url] = expected_model
                return client
            if served_models:
                raise RuntimeError(
                    f"local runtime at {base_url} serves {sorted(served_models)}, expected {expected_model}"
                )
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"failed to switch local runtime chat service at {base_url} to model={expected_model}"
    ) from last_exc
__all__ = [
    "ensure_local_vllm_chat_ready",
    "query_served_model_ids",
    "vllm_chat_result",
]

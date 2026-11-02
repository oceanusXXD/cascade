from __future__ import annotations
import math
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.parse import urlparse

def estimate_text_tokens(text: Any) -> int:
    normalized = str(text or "").strip()
    if not normalized:
        return 0
    by_chars = int(math.ceil(len(normalized) / 4.0))
    by_words = len(normalized.split())
    return max(1, max(by_chars, by_words))

def estimate_texts_tokens(texts: Iterable[Any]) -> int:
    return int(sum(estimate_text_tokens(text) for text in texts))

def estimate_prompt_token_breakdown(
    *,
    system_texts: Iterable[Any] = (),
    user_texts: Iterable[Any] = (),
) -> dict[str, int]:
    system_prompt_tokens = estimate_texts_tokens(system_texts)
    user_prompt_tokens = estimate_texts_tokens(user_texts)
    return {
        "system_prompt_tokens": int(system_prompt_tokens),
        "user_prompt_tokens": int(user_prompt_tokens),
        "prompt_tokens": int(system_prompt_tokens + user_prompt_tokens),
    }

def monotonic_seconds() -> float:
    return float(time.perf_counter())

def elapsed_seconds_since(started_at: float) -> float:
    return round(max(0.0, float(time.perf_counter()) - float(started_at)), 6)

def resolve_model_environment(*, model: str = "", base_url: str = "") -> str:
    resolved_model = str(model or "").strip()
    if resolved_model and Path(resolved_model).exists():
        return "local_model_path"
    raw_base_url = str(base_url or "").strip()
    if raw_base_url:
        parsed = urlparse(raw_base_url if "://" in raw_base_url else f"http://{raw_base_url}")
        host = str(parsed.hostname or "").strip().lower()
        if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
            return "local_openai_compatible"
        return "remote_openai_compatible"
    return "local_model_process"

def model_usage_entry(
    *,
    scope: str,
    stage: str,
    model: str,
    base_url: str = "",
    call_count: int = 0,
    request_count: int = 0,
    input_count: int = 0,
    prompt_tokens: int = 0,
    system_prompt_tokens: int = 0,
    user_prompt_tokens: int = 0,
    completion_tokens: int = 0,
    token_accounting_mode: str = "provider_usage",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_count = int(prompt_tokens or 0)
    system_prompt_count = int(system_prompt_tokens or 0)
    user_prompt_count = int(user_prompt_tokens or 0)
    if prompt_count <= 0 and (system_prompt_count or user_prompt_count):
        prompt_count = system_prompt_count + user_prompt_count
    payload = {
        "scope": str(scope),
        "stage": str(stage),
        "model": str(model or ""),
        "base_url": str(base_url or ""),
        "environment": resolve_model_environment(model=str(model or ""), base_url=str(base_url or "")),
        "call_count": int(call_count or 0),
        "request_count": int(request_count or 0),
        "input_count": int(input_count or 0),
        "prompt_tokens": prompt_count,
        "system_prompt_tokens": system_prompt_count,
        "user_prompt_tokens": user_prompt_count,
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": prompt_count + int(completion_tokens or 0),
        "token_accounting_mode": str(token_accounting_mode or "provider_usage"),
    }
    if extra:
        payload.update(dict(extra))
    return payload

def merge_model_usage_entries(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized_entries = [dict(entry) for entry in entries if isinstance(entry, dict)]
    return {
        "entries": normalized_entries,
        "totals": {
            "call_count": int(sum(int(entry.get("call_count", 0) or 0) for entry in normalized_entries)),
            "request_count": int(sum(int(entry.get("request_count", 0) or 0) for entry in normalized_entries)),
            "input_count": int(sum(int(entry.get("input_count", 0) or 0) for entry in normalized_entries)),
            "prompt_tokens": int(sum(int(entry.get("prompt_tokens", 0) or 0) for entry in normalized_entries)),
            "system_prompt_tokens": int(
                sum(int(entry.get("system_prompt_tokens", 0) or 0) for entry in normalized_entries)
            ),
            "user_prompt_tokens": int(
                sum(int(entry.get("user_prompt_tokens", 0) or 0) for entry in normalized_entries)
            ),
            "completion_tokens": int(
                sum(int(entry.get("completion_tokens", 0) or 0) for entry in normalized_entries)
            ),
            "total_tokens": int(sum(int(entry.get("total_tokens", 0) or 0) for entry in normalized_entries)),
        },
    }

def build_cost_summary(
    *,
    elapsed_seconds: float,
    model_usage: dict[str, Any] | None = None,
    model_call_count: int = 0,
    model_request_count: int = 0,
    model_input_count: int = 0,
    prompt_tokens: int = 0,
    system_prompt_tokens: int = 0,
    user_prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    totals = {}
    if isinstance(model_usage, dict) and isinstance(model_usage.get("totals"), dict):
        totals = dict(model_usage.get("totals", {}) or {})
    call_count = int(totals.get("call_count", model_call_count) or 0)
    request_count = int(totals.get("request_count", model_request_count) or 0)
    input_count = int(totals.get("input_count", model_input_count) or 0)
    prompt_count = int(totals.get("prompt_tokens", prompt_tokens) or 0)
    system_prompt_count = int(totals.get("system_prompt_tokens", system_prompt_tokens) or 0)
    user_prompt_count = int(totals.get("user_prompt_tokens", user_prompt_tokens) or 0)
    if prompt_count <= 0 and (system_prompt_count or user_prompt_count):
        prompt_count = system_prompt_count + user_prompt_count
    completion_count = int(totals.get("completion_tokens", completion_tokens) or 0)
    total_count = int(totals.get("total_tokens", prompt_count + completion_count) or 0)
    return {
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds or 0.0)), 6),
        "model_call_count": call_count,
        "model_request_count": request_count,
        "model_input_count": input_count,
        "prompt_tokens": prompt_count,
        "system_prompt_tokens": system_prompt_count,
        "user_prompt_tokens": user_prompt_count,
        "completion_tokens": completion_count,
        "total_tokens": total_count,
    }

def model_usage_entries_from_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    usage = payload.get("model_usage", payload)
    if not isinstance(usage, dict):
        return []
    return [dict(entry) for entry in list(usage.get("entries", []) or []) if isinstance(entry, dict)]

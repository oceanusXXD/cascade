from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from script.retrieval.settings import ChunkSelectionSettings
from script.retrieval.utils.numeric import safe_int
from script.model_usage import (
    merge_model_usage_entries,
    model_usage_entries_from_payload,
    model_usage_entry,
)

def build_embedding_model_usage(settings: ChunkSelectionSettings, totals: Mapping[str, int]) -> dict[str, Any]:
    return merge_model_usage_entries(
        [
            model_usage_entry(
                scope="script.retrieval.embedding",
                stage="embedding",
                model=str(settings.embedding_model),
                base_url=str(settings.embedding_base_url),
                call_count=int(totals.get("embedding_model_call_count", 0)),
                request_count=int(totals.get("embedding_model_request_count", 0)),
                input_count=int(totals.get("embedding_model_input_count", 0)),
                prompt_tokens=int(totals.get("embedding_prompt_tokens", 0)),
                system_prompt_tokens=int(totals.get("embedding_system_prompt_tokens", 0)),
                user_prompt_tokens=int(totals.get("embedding_user_prompt_tokens", 0)),
                completion_tokens=0,
                token_accounting_mode="estimated_text",
            )
        ]
    )

def build_rerank_model_usage(
    settings: ChunkSelectionSettings,
    *,
    call_count: int,
    request_count: int,
    input_count: int,
    prompt_tokens: int,
    system_prompt_tokens: int,
    user_prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    return merge_model_usage_entries(
        [
            model_usage_entry(
                scope="script.retrieval.rerank",
                stage="rerank",
                model=str(settings.reranker_model),
                base_url=str(settings.reranker_base_url),
                call_count=int(call_count),
                request_count=int(request_count),
                input_count=int(input_count),
                prompt_tokens=int(prompt_tokens),
                system_prompt_tokens=int(system_prompt_tokens),
                user_prompt_tokens=int(user_prompt_tokens),
                completion_tokens=int(completion_tokens),
                token_accounting_mode="estimated_text",
            )
        ]
    )

def source_model_usage(source_meta: Mapping[str, Any]) -> dict[str, Any]:
    retrieval_meta = source_meta.get("source_retrieval_candidates_meta", {})
    entries = [
        *model_usage_entries_from_payload(retrieval_meta if isinstance(retrieval_meta, dict) else {}),
        *model_usage_entries_from_payload(dict(source_meta)),
    ]
    return merge_model_usage_entries(entries)

def meta_int(source_meta: Mapping[str, Any], nested_key: str, key: str) -> int:
    nested = source_meta.get(nested_key, {})
    if isinstance(nested, Mapping) and key in nested:
        return safe_int(nested.get(key), 0)
    return safe_int(source_meta.get(key), 0)

def usage_total_int(source_meta: Mapping[str, Any], key: str) -> int:
    usage = source_meta.get("model_usage", {})
    if not isinstance(usage, Mapping):
        return 0
    totals = usage.get("totals", {})
    if not isinstance(totals, Mapping):
        return 0
    return safe_int(totals.get(key), 0)

def meta_or_usage_int(source_meta: Mapping[str, Any], key: str) -> int:
    if key in source_meta:
        return safe_int(source_meta.get(key), 0)
    return usage_total_int(source_meta, key)
__all__ = [
    "build_embedding_model_usage",
    "build_rerank_model_usage",
    "meta_int",
    "meta_or_usage_int",
    "source_model_usage",
    "usage_total_int",
]

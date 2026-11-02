from __future__ import annotations
from collections.abc import Mapping
from typing import Any

def _row_like_get(row: Any, key: str, default: Any = "") -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)

def first_nonempty_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""

def resolve_default_task_text(row: Any, *, fallback: Any = "") -> str:
    return first_nonempty_text(
        _row_like_get(row, "claim_text", ""),
        _row_like_get(row, "claim", ""),
        _row_like_get(row, "query_text", ""),
        fallback,
    )

def row_task_labels(row: Any) -> tuple[str, str]:
    task_label = first_nonempty_text(_row_like_get(row, "prompt_task_label", "Query"), "Query")
    evidence_label = (
        first_nonempty_text(_row_like_get(row, "prompt_evidence_label", "Document"), "Document")
    )
    return task_label, evidence_label

def resolve_prompt_payload(
    row: Any,
    *,
    default_task_text: Any = "",
    default_evidence_text: Any = "",
) -> dict[str, str]:
    task_label, evidence_label = row_task_labels(row)
    return {
        "task_label": task_label,
        "evidence_label": evidence_label,
        "instruction_text": first_nonempty_text(_row_like_get(row, "prompt_instruction_text", "")),
        "task_text": first_nonempty_text(
            _row_like_get(row, "prompt_task_text", ""),
            default_task_text,
        ),
        "evidence_text": first_nonempty_text(
            _row_like_get(row, "prompt_evidence_text", ""),
            default_evidence_text,
        ),
    }

def resolve_embedding_payload(
    row: Any,
    *,
    default_task_text: Any = "",
    default_evidence_text: Any = "",
) -> dict[str, str]:
    prompt_payload = resolve_prompt_payload(
        row,
        default_task_text=default_task_text,
        default_evidence_text=default_evidence_text,
    )
    return {
        "task_label": prompt_payload["task_label"],
        "evidence_label": prompt_payload["evidence_label"],
        "instruction_text": first_nonempty_text(
            _row_like_get(row, "embedding_instruction_text", ""),
            prompt_payload["instruction_text"],
        ),
        "task_text": first_nonempty_text(
            _row_like_get(row, "embedding_task_text", ""),
            prompt_payload["task_text"],
        ),
        "evidence_text": first_nonempty_text(
            _row_like_get(row, "embedding_evidence_text", ""),
            prompt_payload["evidence_text"],
        ),
    }
__all__ = [
    "first_nonempty_text",
    "resolve_default_task_text",
    "resolve_embedding_payload",
    "resolve_prompt_payload",
    "row_task_labels",
]

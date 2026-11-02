from __future__ import annotations
from typing import Any
SAMPLE_ID_SEPARATOR = ":"

def normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    return text

def build_sample_id(document_id: Any, query_id: Any = "") -> str:
    document_text = normalize_id(document_id)
    query_text = normalize_id(query_id)
    if not document_text:
        raise ValueError("document_id is required")
    if not query_text:
        return document_text
    if SAMPLE_ID_SEPARATOR in document_text or SAMPLE_ID_SEPARATOR in query_text:
        raise ValueError("document_id and query_id must not contain ':'")
    return f"{document_text}{SAMPLE_ID_SEPARATOR}{query_text}"

def split_sample_id(sample_id: Any) -> tuple[str, str]:
    text = normalize_id(sample_id)
    if not text:
        return "", ""
    parts = text.split(SAMPLE_ID_SEPARATOR)
    if len(parts) == 1:
        return parts[0], ""
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"invalid sample_id: {sample_id!r}")
    return parts[0], parts[1]

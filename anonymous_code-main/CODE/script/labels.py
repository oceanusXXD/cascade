from __future__ import annotations
import re
from typing import Any
_BINARY_TEXT_TRUE_VALUES = {"y", "yes", "true", "1"}
_BINARY_TEXT_FALSE_VALUES = {
    "n",
    "no",
    "false",
    "0",
    "refutes",
    "not enough info",
    "not enough information",
}

def normalize_binary_label(value: Any) -> str:
    return normalize_binary_decision_text(value)

def is_binary_label(value: Any) -> bool:
    return bool(normalize_binary_decision_text(value))

def normalize_binary_decision_text(value: Any) -> str:
    raw = str("" if value is None else value).strip()
    if not raw:
        return ""
    candidates = [raw]
    compact = re.sub(r"\s+", "", raw)
    if compact and compact != raw:
        candidates.append(compact)
    for candidate in candidates:
        cleaned = str(candidate).strip()
        previous = None
        while cleaned and cleaned != previous:
            previous = cleaned
            if len(cleaned) >= 2 and cleaned[0] in {'"', "'", "`"} and cleaned[-1] == cleaned[0]:
                cleaned = cleaned[1:-1].strip()
                continue
            break
        for normalized_candidate in (
            cleaned.lower(),
            re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", cleaned).strip().lower(),
        ):
            if normalized_candidate in _BINARY_TEXT_TRUE_VALUES:
                return "yes"
            if normalized_candidate in _BINARY_TEXT_FALSE_VALUES:
                return "no"
    return ""
__all__ = [
    "is_binary_label",
    "normalize_binary_decision_text",
    "normalize_binary_label",
]

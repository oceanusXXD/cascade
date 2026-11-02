from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from script.task_schema import first_nonempty_text, resolve_default_task_text

@dataclass(frozen=True)

class ChunkCandidate:
    text: str
    sentence_indices: tuple[int, ...]

def sentence_windows(sentences: Sequence[str], *, window_size: int = 3) -> list[ChunkCandidate]:
    resolved_window_size = max(1, int(window_size))
    clean_sentences = [str(sentence or "").strip() for sentence in sentences if str(sentence or "").strip()]
    if not clean_sentences:
        return [ChunkCandidate(text="", sentence_indices=tuple())]
    candidates: list[ChunkCandidate] = []
    for start in range(0, len(clean_sentences)):
        window = clean_sentences[start : start + resolved_window_size]
        if not window:
            continue
        candidates.append(
            ChunkCandidate(
                text=" ".join(window).strip(),
                sentence_indices=tuple(range(start, start + len(window))),
            )
        )
    return candidates or [
        ChunkCandidate(
            text=" ".join(clean_sentences).strip(),
            sentence_indices=tuple(range(len(clean_sentences))),
        )
    ]

def resolve_effective_query_text(*, query_text: Any, prompt_task_text: Any = "") -> str:
    return first_nonempty_text(prompt_task_text, query_text)

def resolve_row_effective_query_text(row: Mapping[str, Any]) -> str:
    return first_nonempty_text(
        row.get("prompt_task_text", ""),
        resolve_default_task_text(row),
    )

def effective_query_text(sample: Any) -> str:
    return resolve_effective_query_text(
        query_text=sample.query_text,
        prompt_task_text=sample.prompt_task_text,
    )

def lexical_queries(sample: Any) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    original = effective_query_text(sample)
    if original:
        variants.append(("query", original))
    variants.extend(
        ("query_rewrite", str(item).strip())
        for item in list(sample.query_rewrites or ())
        if str(item).strip()
    )
    return variants or [("query", original)]

def vector_queries(sample: Any) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    original = effective_query_text(sample)
    if original:
        variants.append(("query", original))
    variants.extend(
        ("hyde", str(item).strip())
        for item in list(sample.hyde_hypotheses or ())
        if str(item).strip()
    )
    return variants
__all__ = [
    "ChunkCandidate",
    "effective_query_text",
    "lexical_queries",
    "resolve_effective_query_text",
    "resolve_row_effective_query_text",
    "sentence_windows",
    "vector_queries",
]

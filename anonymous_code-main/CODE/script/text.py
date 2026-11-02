from __future__ import annotations
from typing import Any
SENTENCE_TOKENIZER_NAME = "blingfire"

def split_into_sentences(text: Any) -> list[str]:
    try:
        from blingfire import text_to_sentences
    except ImportError as exc:
        raise RuntimeError(
            " `blingfire` is required for sentence tokenization. Please install it with `pip install blingfire`."
        ) from exc
    normalized = str(text or "").strip()
    if not normalized:
        return []
    segmented = str(text_to_sentences(normalized) or "").strip()
    if not segmented:
        return []
    return [line.strip() for line in segmented.splitlines() if line.strip()]
__all__ = ["SENTENCE_TOKENIZER_NAME", "split_into_sentences"]

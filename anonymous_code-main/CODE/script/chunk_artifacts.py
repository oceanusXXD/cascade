from __future__ import annotations
from pathlib import Path
from typing import Any
from script.io import read_jsonl

def validate_selected_chunk_row(row: dict[str, Any]) -> None:
    required = [
        "sample_id",
        "query_id",
        "document_id",
        "query_text",
        "document_text",
        "selected_chunk_text",
        "strategy",
        "selector_version",
    ]
    missing = [key for key in required if not str(row.get(key, "") or "").strip()]
    if missing:
        raise ValueError(f"selected chunk row missing required keys: {', '.join(missing)}")

def selected_chunks_path(root: str | Path) -> Path:
    return Path(root) / "selected_chunks.jsonl"

def selected_chunks_meta_path(root: str | Path) -> Path:
    return Path(root) / "selected_chunks.meta.json"

def load_selected_chunks(root: str | Path) -> list[dict]:
    return read_jsonl(selected_chunks_path(root))

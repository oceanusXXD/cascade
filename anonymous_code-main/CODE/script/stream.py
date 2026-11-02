from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from script.io import append_jsonl, ensure_directory, write_json

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

class StreamingRunWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = ensure_directory(root)
    def write_event(self, event: str, **payload: Any) -> Path:
        row = {"event": str(event), "timestamp": _utcnow_iso(), **payload}
        return append_jsonl(self.root / "events.jsonl", row)
    def write_stage_row(self, stage_name: str, row: dict[str, Any]) -> Path:
        return append_jsonl(self.root / f"{stage_name}.jsonl", dict(row))
    def write_progress(self, *, stage: str, completed: int, total: int, **payload: Any) -> Path:
        progress = {
            "stage": str(stage),
            "completed": int(completed),
            "total": int(total),
            "timestamp": _utcnow_iso(),
            **payload,
        }
        return write_json(self.root / "progress.json", progress)
    def write_summary(self, payload: dict[str, Any]) -> Path:
        return write_json(self.root / "summary.json", payload)

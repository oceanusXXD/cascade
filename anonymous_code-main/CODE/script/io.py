from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable

def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_optional_json_object(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    payload = read_json(file_path)
    return dict(payload) if isinstance(payload, dict) else {}

def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target

def append_jsonl(path: str | Path, row: dict[str, Any]) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target

def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target

def iter_jsonl_objects(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"jsonl row must be an object: {path}")
            yield payload

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl_objects(path))

def read_jsonl_lenient(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows

def count_jsonl_rows(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return int(sum(1 for line in handle if line.strip()))

def remove_existing_files(root: str | Path, names: Iterable[str]) -> None:
    base = Path(root)
    for name in names:
        path = base / str(name)
        if path.exists():
            path.unlink()
__all__ = [
    "append_jsonl",
    "count_jsonl_rows",
    "ensure_directory",
    "iter_jsonl_objects",
    "read_optional_json_object",
    "read_json",
    "read_jsonl",
    "read_jsonl_lenient",
    "remove_existing_files",
    "write_json",
    "write_jsonl",
]

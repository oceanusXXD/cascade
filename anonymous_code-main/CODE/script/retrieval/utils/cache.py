from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)

class QueryGenerationCacheSpec:
    kind: str
    display_name: str
    filename: str
    value_field: str
_CACHE_SPECS = {
    "hyde": QueryGenerationCacheSpec(
        kind="hyde",
        display_name="HyDE",
        filename="hyde.json",
        value_field="hypotheses",
    ),
    "query_rewrite": QueryGenerationCacheSpec(
        kind="query_rewrite",
        display_name="query rewrite",
        filename="query_rewrite.json",
        value_field="rewrites",
    ),
}

def normalize_query_cache_key(query: str) -> str:
    return " ".join(str(query or "").strip().lower().split())

def _dedupe_texts(items: Any) -> tuple[str, ...]:
    if not isinstance(items, list):
        return ()
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item or "").strip().split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return tuple(deduped)

def _cache_spec(kind: str) -> QueryGenerationCacheSpec:
    normalized_kind = str(kind or "").strip().lower()
    spec = _CACHE_SPECS.get(normalized_kind)
    if spec is None:
        raise ValueError(f"unsupported query generation cache kind: {kind!r}")
    return spec

def _candidate_cache_dirs(queries_path: str | Path) -> list[Path]:
    query_dir = Path(queries_path).resolve().parent
    return [(query_dir / "query_generation_cache").resolve()]

def _resolve_existing_cache_path(kind: str, queries_path: str | Path) -> Path | None:
    spec = _cache_spec(kind)
    for directory in _candidate_cache_dirs(queries_path):
        path = directory / spec.filename
        if path.exists():
            return path
    return None

def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to load query generation cache: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid query generation cache payload in {path}: expected object")
    return payload

@dataclass(frozen=True)

class QueryGenerationCacheLookup:
    spec: QueryGenerationCacheSpec
    path: Path
    entries: dict[str, dict[str, Any]]
    def get(self, query: str) -> tuple[str, ...]:
        key = normalize_query_cache_key(query)
        row = self.entries.get(key)
        if not isinstance(row, dict):
            raise RuntimeError(f"{self.display_name} cache entry missing for query={query!r}")
        values = _dedupe_texts(row.get(self.spec.value_field))
        if not values:
            raise RuntimeError(f"{self.spec.display_name} cache entry contains no valid values for query={query!r}")
        return values
    @property
    def display_name(self) -> str:
        return self.spec.display_name

def load_query_generation_cache(kind: str, *, queries_path: str | Path) -> QueryGenerationCacheLookup:
    spec = _cache_spec(kind)
    path = _resolve_existing_cache_path(spec.kind, queries_path)
    if path is None:
        searched = ", ".join(str(directory / spec.filename) for directory in _candidate_cache_dirs(queries_path))
        raise RuntimeError(f"{spec.display_name} cache file not found; searched: {searched}")
    payload = _load_payload(path)
    raw_entries = payload.get("entries", {})
    if not isinstance(raw_entries, dict):
        raise RuntimeError(f"invalid query generation cache entries in {path}: expected object")
    entries = {
        normalize_query_cache_key(str(key)): value
        for key, value in raw_entries.items()
        if normalize_query_cache_key(str(key)) and isinstance(value, dict)
    }
    return QueryGenerationCacheLookup(
        spec=spec,
        path=path,
        entries=entries,
    )

from __future__ import annotations
import json
import re
from script.labels import normalize_binary_decision_text
_STEP3_BATCH_LABEL_RE = re.compile(
    r"^\s*(?:\[(?P<bracket_idx>\d+)\]|(?P<plain_idx>\d+))\s*[:.-]?\s*(?P<label>[A-Za-z0-9]+)\s*$",
    flags=re.IGNORECASE,
)


def extract_json_candidate(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return ""
    return raw[start : end + 1]

def parse_step3_batch_output(text: str, document_count: int, *, uses_chunk_json_protocol: bool) -> list[str]:
    if uses_chunk_json_protocol:
        candidate = extract_json_candidate(text)
        if not candidate:
            raise ValueError("missing_batch_json")
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("batch_json_not_object")
        if payload.get("result") == "all_no":
            return ["no" for _ in range(int(document_count))]
        if payload.get("result") != "has_yes":
            raise ValueError("unexpected_batch_json_result")
        yes_indices_raw = payload.get("yes_indices", [])
        if not isinstance(yes_indices_raw, list):
            raise ValueError("yes_indices_not_list")
        yes_indices = {int(idx) for idx in yes_indices_raw if isinstance(idx, int)}
        return ["yes" if idx in yes_indices else "no" for idx in range(int(document_count))]
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty_batch_response")
    if raw.startswith("```"):
        rows = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(rows).strip()
    observed_indices: list[int] = []
    labels: list[str] = []
    for row in raw.replace(";", "\n").splitlines():
        item = str(row or "").strip()
        if not item:
            continue
        match = _STEP3_BATCH_LABEL_RE.fullmatch(item)
        if match is None:
            continue
        raw_idx = match.group("bracket_idx") or match.group("plain_idx") or ""
        parsed_idx = int(raw_idx)
        label = normalize_binary_decision_text(match.group("label") or "")
        if label not in {"yes", "no"}:
            raise ValueError(f"invalid_batch_label:{item!r}")
        observed_indices.append(parsed_idx)
        labels.append(label)
    expected = list(range(1, int(document_count) + 1))
    if observed_indices != expected:
        raise ValueError(f"unexpected_batch_indices:{observed_indices!r}")
    return labels
__all__ = [
    "extract_json_candidate",
    "parse_step3_batch_output",
]

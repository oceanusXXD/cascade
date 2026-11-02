from __future__ import annotations
import random
from typing import Any, Iterable
import numpy as np
from script.labels import is_binary_label, normalize_binary_label

def sample_to_result_row(sample: Any) -> dict[str, Any]:
    return {
        "sample_id": str(getattr(sample, "sample_id", "") or ""),
        "query": str(getattr(sample, "query", "") or ""),
        "document_text": str(getattr(sample, "text", "") or ""),
        "prediction": normalize_binary_label(getattr(sample, "prediction", "")),
        "label": normalize_binary_label(getattr(sample, "label", "")),
        "action": str(getattr(getattr(sample, "action", ""), "value", getattr(sample, "action", "")) or ""),
        "source": str(getattr(getattr(sample, "source", ""), "value", getattr(sample, "source", "")) or ""),
        "path": list(getattr(sample, "path", []) or []),
        "defer_reason": str(getattr(sample, "defer_reason", "") or ""),
    }

def select_calibration_samples_srswor(samples: list[Any], *, calibration_ratio: float, seed: int) -> list[Any]:
    if calibration_ratio <= 0.0:
        return []
    population = list(samples)
    if not population:
        return []
    target = max(1, int(round(len(population) * float(calibration_ratio))))
    target = min(int(target), len(population))
    rng = random.Random(int(seed))
    selected = rng.sample(population, target)
    rng.shuffle(selected)
    return selected

def value_distribution(values: Iterable[str]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip() or "unknown"
        dist[key] = int(dist.get(key, 0)) + 1
    return dict(sorted(dist.items(), key=lambda item: item[0]))

def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }

def _sample_signal_quality_key(sample: Any) -> str:
    if bool(getattr(sample, "step1_parse_failed", False)):
        return "parse_failed"
    return str(getattr(sample, "step1_signal_quality", "") or "").strip() or "unknown"

def signal_quality_distribution_from_samples(
    samples: Iterable[Any],
    *,
    labeled_only: bool = False,
) -> dict[str, int]:
    values: list[str] = []
    for sample in samples:
        if labeled_only and not is_binary_label(getattr(sample, "label", "")):
            continue
        values.append(_sample_signal_quality_key(sample))
    return value_distribution(values)

def neighbor_support_summary_from_samples(
    samples: Iterable[Any],
    *,
    labeled_only: bool = False,
    support_attr: str = "step1_neighbor_support",
    prediction_attr: str = "stage1_prediction",
) -> dict[str, float]:
    values: list[float] = []
    for sample in samples:
        if labeled_only and not is_binary_label(getattr(sample, "label", "")):
            continue
        if is_binary_label(getattr(sample, prediction_attr, "")):
            values.append(float(getattr(sample, support_attr, 0.0) or 0.0))
    return numeric_summary(values)
__all__ = [
    "neighbor_support_summary_from_samples",
    "numeric_summary",
    "sample_to_result_row",
    "select_calibration_samples_srswor",
    "signal_quality_distribution_from_samples",
    "value_distribution",
]

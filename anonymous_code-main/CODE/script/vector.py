from __future__ import annotations
import numpy as np

def normalize_vectors(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return (arr / norms).astype(np.float32)
__all__ = ["normalize_vectors"]

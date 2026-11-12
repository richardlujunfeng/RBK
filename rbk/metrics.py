from __future__ import annotations

import numpy as np


def cluster_sizes(ids: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(ids.astype(np.int64), minlength=k).astype(np.int64)


def cv_sizes(counts: np.ndarray) -> float:
    c = counts.astype(np.float64)
    m = float(c.mean())
    if m <= 0:
        return 0.0
    return float(c.std() / m)


def sse(X: np.ndarray, C: np.ndarray, ids: np.ndarray) -> float:
    diff = X - C[ids]
    return float(np.einsum("ij,ij->", diff, diff))

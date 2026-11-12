"""Phase (iv): damped centroid update (+ helpers)."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def cluster_means(
    X: np.ndarray,
    ids: np.ndarray,
    K: int,
    C_fallback: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-cluster means; reseed empty clusters from data."""
    counts = np.bincount(ids, minlength=K).astype(np.int64)
    D = X.shape[1]
    sums = np.zeros((K, D), dtype=np.float64)
    np.add.at(sums, ids, X.astype(np.float64, copy=False))
    C_mean = C_fallback.astype(np.float32).copy()
    nz = counts > 0
    C_mean[nz] = (sums[nz] / counts[nz, None]).astype(np.float32)
    empty = np.flatnonzero(~nz)
    if empty.size:
        pick = rng.choice(X.shape[0], size=empty.size, replace=False)
        C_mean[empty] = X[pick].astype(np.float32)
    return C_mean, counts


def damped_centroid_update(
    C_old: np.ndarray,
    C_mean: np.ndarray,
    counts: np.ndarray,
    *,
    t: int,
    eta0: float = 1.0,
    beta: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    mu <- (1-eta_t)*mu + eta_t*mean,  eta_t = eta0 * exp(-beta * t).

    Default (eta0=1, beta=0) recovers standard Lloyd. Nonzero beta is used in
    the damped-centroid ablation.
    """
    eta = float(eta0 * math.exp(-beta * t))
    eta = min(max(eta, 0.0), 1.0)
    if abs(eta - 1.0) < 1e-12:
        C_new = C_mean.astype(np.float32).copy()
    else:
        C_new = ((1.0 - eta) * C_old + eta * C_mean).astype(np.float32)
        empty = counts <= 0
        C_new[empty] = C_mean[empty]
    return C_new, eta

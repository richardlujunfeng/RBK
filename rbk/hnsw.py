"""Phase (i): candidate retrieval via dynamic HNSW over centroids."""
from __future__ import annotations

from typing import Optional, Tuple

import faiss
import numpy as np


def build_centroid_hnsw(
    C: np.ndarray,
    *,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
):
    C = np.ascontiguousarray(C.astype(np.float32))
    index = faiss.IndexHNSWFlat(C.shape[1], int(M))
    index.hnsw.efConstruction = int(max(ef_construction, 4 * M))
    index.hnsw.efSearch = int(ef_search)
    index.add(C)
    return index


def search_topk(
    index: faiss.Index,
    X: np.ndarray,
    L: int,
    *,
    batch: int = 65536,
    ef_search: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dist[N,L], idx[N,L]) for L nearest centroids."""
    if ef_search is not None and hasattr(index, "hnsw"):
        index.hnsw.efSearch = int(ef_search)
    N = X.shape[0]
    dist = np.empty((N, L), dtype=np.float32)
    idx = np.empty((N, L), dtype=np.int32)
    X = np.ascontiguousarray(X.astype(np.float32))
    for s in range(0, N, batch):
        e = min(s + batch, N)
        d, ix = index.search(X[s:e], L)
        dist[s:e] = d
        idx[s:e] = ix
    return dist, idx


def drifted_mask(C_old: np.ndarray, C_new: np.ndarray, delta: float = 1e-2) -> np.ndarray:
    """Relative centroid drift: ||c'-c|| / (||c||+eps) > delta."""
    d = np.linalg.norm((C_new - C_old).astype(np.float64), axis=1)
    nrm = np.linalg.norm(C_old.astype(np.float64), axis=1) + 1e-12
    return d / nrm > float(delta)

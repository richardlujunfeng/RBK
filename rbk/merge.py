"""Post-loop cleanup: merge fragmented micro-clusters into under-full receivers."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def merge_tiny_into_underfull(
    X: np.ndarray,
    centroids: np.ndarray,
    ids: np.ndarray,
    counts: np.ndarray,
    n_bar: float,
    *,
    tiny_ratio: float = 0.25,
    min_tiny: int = 1,
    refresh_centroids: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Dissolve tiny clusters (n_j < tiny_ratio * n_bar) into nearest receivers
    with current size strictly below n_bar when possible.
    """
    ids = ids.copy()
    counts = counts.astype(np.int64).copy()
    C = centroids.astype(np.float32).copy()
    K = C.shape[0]
    n_bar_i = float(n_bar)
    n_tiny = max(min_tiny, int(tiny_ratio * n_bar_i))

    tiny = set(int(j) for j in np.flatnonzero(counts < n_tiny))
    if not tiny:
        return C, ids, counts, dict(
            n_tiny_thresh=n_tiny, n_tiny_clusters=0, n_moved=0,
            n_empty_after=int((counts == 0).sum()),
        )

    move_idx = np.flatnonzero(np.isin(ids, list(tiny)))
    n_tiny_clusters = len(tiny)
    for j in tiny:
        counts[j] = 0

    recv_all = np.array([k for k in range(K) if k not in tiny], dtype=np.int32)
    if recv_all.size == 0:
        return C, ids, counts, dict(
            n_tiny_thresh=n_tiny, n_tiny_clusters=n_tiny_clusters,
            n_moved=0, n_empty_after=int((counts == 0).sum()),
        )

    C_recv = C[recv_all]
    c2 = np.einsum("ij,ij->i", C_recv, C_recv)
    n_moved = 0
    batch = 8192
    for s in range(0, move_idx.size, batch):
        e = min(s + batch, move_idx.size)
        xb = X[move_idx[s:e]]
        x2 = np.einsum("ij,ij->i", xb, xb)
        dmat = x2[:, None] + c2[None, :] - 2.0 * (xb @ C_recv.T)
        for row, pi in enumerate(move_idx[s:e]):
            order = np.argsort(dmat[row])
            placed = False
            for j in order:
                k = int(recv_all[int(j)])
                if counts[k] < n_bar_i:
                    ids[int(pi)] = k
                    counts[k] += 1
                    n_moved += 1
                    placed = True
                    break
            if not placed:
                # fallback: nearest receiver regardless of n_bar
                k = int(recv_all[int(order[0])])
                ids[int(pi)] = k
                counts[k] += 1
                n_moved += 1

    if refresh_centroids:
        nz = counts > 0
        for k in np.flatnonzero(nz):
            members = X[ids == int(k)]
            if members.size:
                C[int(k)] = members.mean(axis=0).astype(np.float32)

    return C, ids, counts, dict(
        n_tiny_thresh=n_tiny,
        n_tiny_clusters=n_tiny_clusters,
        n_moved=n_moved,
        n_empty_after=int((counts == 0).sum()),
    )

"""Phase (ii): high-regret-first capacity-constrained greedy assignment."""
from __future__ import annotations

import numpy as np


def fill_capacity(
    order: np.ndarray,
    neigh: np.ndarray,
    rem: np.ndarray,
) -> np.ndarray:
    """
    Greedy assign points in ``order`` to the first neighbor with remaining capacity.
    Unassigned points keep id = -1.
    """
    n, L = neigh.shape
    ids = np.full(n, -1, dtype=np.int32)
    rem = rem.astype(np.int32, copy=True)
    for i in order:
        i = int(i)
        for t in range(L):
            c = int(neigh[i, t])
            if rem[c] > 0:
                ids[i] = c
                rem[c] -= 1
                break
    return ids


def repair_unassigned(
    X: np.ndarray,
    C: np.ndarray,
    ids: np.ndarray,
    rem: np.ndarray,
) -> np.ndarray:
    """Assign leftovers to nearest open centroids, then force-fill if needed."""
    ids = ids.copy()
    rem = rem.astype(np.int32, copy=True)
    left = np.flatnonzero(ids < 0)
    if left.size == 0:
        return ids

    open_idx = np.flatnonzero(rem > 0)
    if open_idx.size == 0:
        ids[left] = 0
        return ids

    C_open = C[open_idx]
    c2 = np.einsum("ij,ij->i", C_open, C_open)
    for s in range(0, left.size, 2048):
        e = min(s + 2048, left.size)
        xb = X[left[s:e]]
        x2 = np.einsum("ij,ij->i", xb, xb)
        dmat = x2[:, None] + c2[None, :] - 2.0 * (xb @ C_open.T)
        take = min(8, dmat.shape[1])
        kk = min(max(take - 1, 0), dmat.shape[1] - 1)
        nn = np.argpartition(dmat, kk, axis=1)[:, :take]
        for row_i, pi in enumerate(left[s:e]):
            for j in nn[row_i, np.argsort(dmat[row_i, nn[row_i]])]:
                c = int(open_idx[int(j)])
                if rem[c] > 0:
                    ids[int(pi)] = c
                    rem[c] -= 1
                    break

    still = np.flatnonzero(ids < 0)
    for pi in still:
        c = int(np.argmax(rem))
        if rem[c] <= 0:
            ids[int(pi)] = 0
        else:
            ids[int(pi)] = c
            rem[c] -= 1
    return ids


def high_regret_capacity_assign(
    X: np.ndarray,
    C: np.ndarray,
    dist: np.ndarray,
    neigh: np.ndarray,
    cap: int,
) -> np.ndarray:
    """
    Sort by large margin / regret first: Δ = d_(2) - d_(1), then capacity fill.
    """
    L = neigh.shape[1]
    margin = dist[:, min(1, L - 1)] - dist[:, 0]
    order = np.argsort(-margin, kind="mergesort")
    rem = np.full(C.shape[0], int(cap), dtype=np.int32)
    ids = fill_capacity(order, neigh, rem)
    return repair_unassigned(X, C, ids, rem)

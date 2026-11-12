from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .capacity import fill_capacity
from .metrics import cluster_sizes, cv_sizes, sse


def _topk_flat(X: np.ndarray, C: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray]:
    x2 = np.einsum("ij,ij->i", X, X)
    c2 = np.einsum("ij,ij->i", C, C)
    d = x2[:, None] + c2[None, :] - 2.0 * (X @ C.T)
    L = int(min(L, C.shape[0]))
    part = np.argpartition(d, L - 1, axis=1)[:, :L]
    rows = np.arange(X.shape[0])[:, None]
    vals = d[rows, part]
    order = np.argsort(vals, axis=1)
    neigh = part[rows, order].astype(np.int32)
    dist = vals[rows, order].astype(np.float32)
    return neigh, dist


def _repair_unassigned(X, C, ids, rem):
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
        nn = np.argpartition(dmat, take - 1, axis=1)[:, :take]
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


def constrained_swap(
    X: np.ndarray,
    C: np.ndarray,
    ids: np.ndarray,
    neigh: np.ndarray,
    dist: np.ndarray,
    top_frac: float = 0.005,
    top_cap: int = 8000,
    partners: int = 16,
) -> np.ndarray:
    ids = ids.copy()
    N = ids.shape[0]
    prefer = neigh[:, 0].astype(np.int32)
    regret_proxy = dist[:, min(1, dist.shape[1] - 1)] - dist[:, 0]
    cand = np.flatnonzero(prefer != ids)
    if cand.size == 0:
        return ids
    score = regret_proxy[cand]
    order = np.argsort(-score, kind="mergesort")
    n_take = min(int(top_cap), max(1, int(top_frac * N)), cand.size)
    cand = cand[order[:n_take]]

    need = set(int(prefer[i]) for i in cand)
    buckets: Dict[int, List[int]] = {k: [] for k in need}
    lim = int(partners)
    filled = 0
    target = len(need) * lim
    for p in range(N):
        j = int(ids[p])
        if j in buckets and len(buckets[j]) < lim:
            buckets[j].append(p)
            filled += 1
            if filled >= target:
                break

    for i in cand:
        i = int(i)
        pref = int(prefer[i])
        cur = int(ids[i])
        if pref == cur:
            continue
        partners_i = buckets.get(pref)
        if not partners_i:
            continue
        ci = C[cur]
        cp = C[pref]
        xi = X[i]
        di_cur = float(np.dot(xi - ci, xi - ci))
        di_pref = float(np.dot(xi - cp, xi - cp))
        best_p = -1
        best_gain = 0.0
        for p2 in partners_i:
            if p2 == i:
                continue
            x2 = X[p2]
            d2_pref = float(np.dot(x2 - cp, x2 - cp))
            d2_cur = float(np.dot(x2 - ci, x2 - ci))
            gain = (di_cur + d2_pref) - (di_pref + d2_cur)
            if gain > best_gain:
                best_gain = gain
                best_p = int(p2)
        if best_p < 0 or best_gain <= 1e-9:
            continue
        ids[i] = pref
        ids[best_p] = cur
    return ids


def merge_tiny(
    X: np.ndarray,
    C: np.ndarray,
    ids: np.ndarray,
    counts: np.ndarray,
    n_bar: float,
    tiny_ratio: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = ids.copy()
    counts = counts.astype(np.int64).copy()
    C = C.astype(np.float32).copy()
    K = C.shape[0]
    n_tiny = max(1, int(tiny_ratio * float(n_bar)))
    tiny = set(int(j) for j in np.flatnonzero(counts < n_tiny))
    if not tiny:
        return C, ids, counts
    move_idx = np.flatnonzero(np.isin(ids, list(tiny)))
    for j in tiny:
        counts[j] = 0
    recv_all = np.array([k for k in range(K) if k not in tiny], dtype=np.int32)
    if recv_all.size == 0:
        return C, ids, counts
    C_recv = C[recv_all]
    c2 = np.einsum("ij,ij->i", C_recv, C_recv)
    cap = np.maximum(float(n_bar) - counts.astype(np.float64), 0.0)
    for s in range(0, move_idx.size, 8192):
        e = min(s + 8192, move_idx.size)
        xb = X[move_idx[s:e]]
        x2 = np.einsum("ij,ij->i", xb, xb)
        dmat = x2[:, None] + c2[None, :] - 2.0 * (xb @ C_recv.T)
        order = np.argsort(dmat, axis=1)
        for row_i, pi in enumerate(move_idx[s:e]):
            assigned = False
            for j in order[row_i]:
                c = int(recv_all[int(j)])
                if cap[c] > 0:
                    ids[int(pi)] = c
                    counts[c] += 1
                    cap[c] -= 1.0
                    assigned = True
                    break
            if not assigned:
                c = int(recv_all[int(order[row_i, 0])])
                ids[int(pi)] = c
                counts[c] += 1
    nz = counts > 0
    if nz.any():
        for j in np.flatnonzero(nz):
            members = X[ids == j]
            if members.shape[0] > 0:
                C[j] = members.mean(axis=0).astype(np.float32)
    return C, ids, counts


def _update_centroids(X, ids, C, rng):
    K = C.shape[0]
    counts = cluster_sizes(ids, K)
    C_new = C.copy()
    for j in range(K):
        m = ids == j
        if np.any(m):
            C_new[j] = X[m].mean(axis=0)
        else:
            C_new[j] = X[int(rng.integers(0, X.shape[0]))]
    return C_new.astype(np.float32), counts


def fit_rbk(
    X: np.ndarray,
    k: int,
    alpha: float = 1.2,
    n_iter: int = 12,
    L: int = 24,
    do_swap: bool = True,
    do_merge: bool = True,
    swap_top_frac: float = 0.005,
    swap_top_cap: int = 8000,
    swap_partners: int = 16,
    seed: int = 0,
    C0: Optional[np.ndarray] = None,
) -> Dict:
    X = np.ascontiguousarray(X, dtype=np.float32)
    N, D = X.shape
    rng = np.random.default_rng(seed)
    if C0 is None:
        idx = rng.choice(N, size=k, replace=False)
        C = X[idx].copy()
    else:
        C = np.ascontiguousarray(C0.astype(np.float32).copy())
    n_bar = N / float(k)
    cap = int(np.floor(alpha * n_bar))
    if cap * k < N:
        cap = int(np.ceil(N / float(k)))

    history = []
    ids = np.zeros(N, dtype=np.int32)
    for it in range(n_iter):
        neigh, dist = _topk_flat(X, C, L)
        margin = dist[:, min(1, L - 1)] - dist[:, 0]
        order = np.argsort(-margin, kind="mergesort")
        rem = np.full(k, int(cap), dtype=np.int32)
        ids = fill_capacity(order, neigh, rem)
        ids = _repair_unassigned(X, C, ids, rem)
        if do_swap:
            ids = constrained_swap(
                X, C, ids, neigh, dist,
                top_frac=swap_top_frac, top_cap=swap_top_cap, partners=swap_partners,
            )
        C, counts = _update_centroids(X, ids, C, rng)
        if do_merge and it == n_iter - 1:
            C, ids, counts = merge_tiny(X, C, ids, counts, n_bar)
        history.append(dict(
            iter=it + 1,
            SSE=sse(X, C, ids),
            cv=cv_sizes(counts),
            max_n=int(counts.max()),
            min_n=int(counts.min()),
        ))
    return dict(centroids=C, labels=ids, history=history, alpha=alpha, cap=cap)

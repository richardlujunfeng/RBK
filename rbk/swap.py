"""Phase (iii): localized constrained 1-1 swap refinement."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def constrained_swap(
    X: np.ndarray,
    C: np.ndarray,
    ids: np.ndarray,
    counts: np.ndarray,
    neigh: np.ndarray,
    dist: np.ndarray,
    cap: int,
    *,
    top_frac: float = 0.005,
    top_cap: int = 8000,
    partners: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Capacity-preserving swaps for high-regret mismatched points.

    Candidates: top min(floor(top_frac * N), top_cap) points with largest
    margin among those not assigned to their preferred (nearest) centroid.
    For each preferred centroid, at most ``partners`` current residents are
    considered as swap partners. A swap is accepted only if it reduces SSE.
    """
    ids = ids.copy()
    counts = counts.astype(np.int64).copy()
    N = ids.shape[0]
    prefer = neigh[:, 0].astype(np.int32)
    regret_proxy = dist[:, min(1, dist.shape[1] - 1)] - dist[:, 0]
    mask = prefer != ids
    cand = np.flatnonzero(mask)
    if cand.size == 0:
        return ids, counts

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
        if pref == cur or counts[pref] <= 0:
            continue
        partner_list = buckets.get(pref)
        if not partner_list:
            continue
        ci = C[cur]
        cp = C[pref]
        xi = X[i]
        di_cur = float(np.dot(xi - ci, xi - ci))
        di_pref = float(np.dot(xi - cp, xi - cp))
        best_p = -1
        best_gain = 0.0
        for p2 in partner_list:
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
        # capacity-preserving 1-1 exchange
        ids[i] = pref
        ids[best_p] = cur
        # counts unchanged in total per cluster under pure swap of two points
        # (both clusters keep size)
    counts = np.bincount(ids, minlength=C.shape[0]).astype(np.int64)
    return ids, counts

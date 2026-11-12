"""
RBK (Regret-aware Balanced K-means).

Per Lloyd iteration (capacity n_j <= U without network flows):
  (i)   candidate retrieval + regret extraction via dynamic HNSW
  (ii)  high-regret-first capacity-constrained greedy assignment
  (iii) localized constrained swap refinement
  (iv)  damped centroid update with drift-aware incremental HNSW maintenance
Final cleanup merges fragmented micro-clusters before indexing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .assign import high_regret_capacity_assign
from .hnsw import build_centroid_hnsw, drifted_mask, search_topk
from .merge import merge_tiny_into_underfull
from .swap import constrained_swap
from .update import cluster_means, damped_centroid_update


@dataclass
class RBKConfig:
    # IVF / Lloyd
    K: int = 4096
    L: int = 24
    alpha: float = 1.2          # U = floor(alpha * N/K)
    n_iters: int = 12
    seed: int = 0

    # HNSW (centroid graph)
    hnsw_M: int = 32
    ef_construction: int = 200
    ef_search: int = 64

    # Drift threshold for rebuilding the centroid HNSW
    drift_delta: float = 1e-2

    # Constrained swap (defaults match paper)
    swap_top_frac: float = 0.005
    swap_top_cap: int = 8000
    swap_partners: int = 16
    do_swap: bool = True

    # Damped centroid update: eta_t = eta0 * exp(-beta * t)
    eta0: float = 1.0
    beta: float = 0.1

    # Post-merge micro-cluster cleanup
    do_post_merge: bool = True
    tiny_ratio: float = 0.25

    # Book-keeping
    log_every: int = 1

    extra: Dict[str, Any] = field(default_factory=dict)


def _sum_sq(X: np.ndarray, C: np.ndarray, ids: np.ndarray) -> float:
    # batch to limit memory
    total = 0.0
    bs = 65536
    for s in range(0, X.shape[0], bs):
        e = min(s + bs, X.shape[0])
        diff = X[s:e] - C[ids[s:e]]
        total += float(np.einsum("ij,ij->", diff, diff))
    return total


def _cv(counts: np.ndarray) -> float:
    m = float(counts.mean()) if counts.size else 0.0
    return float(counts.astype(np.float64).std() / m) if m > 0 else 0.0


def train_rbk(
    X: np.ndarray,
    C0: np.ndarray,
    cfg: Optional[RBKConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Run RBK on ``X`` starting from centroids ``C0``.

    Returns (centroids, ids, counts, history).
    """
    cfg = cfg or RBKConfig()
    X = np.ascontiguousarray(X.astype(np.float32))
    C = np.ascontiguousarray(C0.astype(np.float32).copy())
    N, D = X.shape
    K = int(C.shape[0])
    assert K == cfg.K or cfg.K <= 0 or K > 0
    K = C.shape[0]

    n_bar = N / float(K)
    cap = int(np.floor(cfg.alpha * n_bar))
    if cap * K < N:
        cap = int(np.ceil(N / float(K)))
    U = float(cap)

    rng = np.random.default_rng(cfg.seed)
    L = int(min(cfg.L, K))
    ef = int(max(cfg.ef_search, L))

    # --- initial centroid graph ---
    index = build_centroid_hnsw(
        C, M=cfg.hnsw_M, ef_construction=cfg.ef_construction, ef_search=ef,
    )

    history: List[Dict[str, Any]] = []
    ids = np.zeros(N, dtype=np.int32)
    counts = np.zeros(K, dtype=np.int64)

    for it in range(cfg.n_iters):
        t = it + 1

        # ============================================================
        # (i) candidate retrieval + regret extraction via dynamic HNSW
        # ============================================================
        dist, neigh = search_topk(index, X, L, ef_search=ef)
        # regret / margin proxy Δ = d_(2) - d_(1)  (used in phase ii / iii)

        # ============================================================
        # (ii) high-regret-first capacity-constrained greedy assignment
        # ============================================================
        ids = high_regret_capacity_assign(X, C, dist, neigh, cap)
        counts = np.bincount(ids, minlength=K).astype(np.int64)

        # ============================================================
        # (iii) localized constrained swap refinement
        # ============================================================
        if cfg.do_swap:
            ids, counts = constrained_swap(
                X, C, ids, counts, neigh, dist, cap,
                top_frac=cfg.swap_top_frac,
                top_cap=cfg.swap_top_cap,
                partners=cfg.swap_partners,
            )

        # ============================================================
        # (iv) damped centroid update + drift-aware HNSW maintenance
        # ============================================================
        C_mean, counts = cluster_means(X, ids, K, C, rng)
        C_new, eta = damped_centroid_update(
            C, C_mean, counts, t=t, eta0=cfg.eta0, beta=cfg.beta,
        )
        drift = drifted_mask(C, C_new, delta=cfg.drift_delta)
        if int(drift.sum()) > 0:
            # Faiss HNSW has no cheap in-place centroid edit → rebuild
            index = build_centroid_hnsw(
                C_new, M=cfg.hnsw_M,
                ef_construction=cfg.ef_construction, ef_search=ef,
            )
        C = C_new

        sse = _sum_sq(X, C, ids)
        cv = _cv(counts)
        row = dict(
            iter=t, SSE=sse, cv=cv, eta=eta,
            max_n=int(counts.max()), min_n=int(counts.min()),
            empty=int((counts == 0).sum()),
            n_drifted=int(drift.sum()),
            max_ratio=float(counts.max()) / max(n_bar, 1e-12),
            U=U, cap=cap,
        )
        history.append(row)
        if cfg.log_every > 0 and (t % cfg.log_every == 0 or t == 1):
            print(
                f"[RBK] it={t:02d} eta={eta:.3f} SSE={sse:.6e} cv={cv:.4f} "
                f"drifted={row['n_drifted']}",
                flush=True,
            )

    # ================================================================
    # Final cleanup: merge fragmented micro-clusters before indexing
    # ================================================================
    if cfg.do_post_merge:
        C, ids, counts, stats = merge_tiny_into_underfull(
            X, C, ids, counts, float(U),
            tiny_ratio=cfg.tiny_ratio, refresh_centroids=True,
        )
        if history:
            history[-1]["post_merge"] = stats
        print(
            f"[RBK] post-merge tiny_clusters={stats.get('n_tiny_clusters')} "
            f"moved={stats.get('n_moved')}",
            flush=True,
        )

    return C, ids, counts, history


def config_dict(cfg: RBKConfig) -> Dict[str, Any]:
    return asdict(cfg)

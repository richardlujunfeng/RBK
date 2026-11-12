#!/usr/bin/env python3
"""Smoke test: RBK on a small synthetic Gaussian mixture."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbk import RBKConfig, train_rbk  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    K, N, D = 32, 5000, 16
    centers = rng.normal(size=(K, D)).astype(np.float32)
    ids0 = rng.integers(0, K, size=N)
    X = centers[ids0] + 0.15 * rng.normal(size=(N, D)).astype(np.float32)
    C0 = centers + 0.05 * rng.normal(size=(K, D)).astype(np.float32)

    cfg = RBKConfig(
        K=K, L=8, alpha=1.2, n_iters=5,
        hnsw_M=16, ef_construction=64, ef_search=32,
        swap_top_frac=0.01, swap_top_cap=200, swap_partners=8,
        eta0=1.0, beta=0.1,
        do_post_merge=True, log_every=1,
    )
    C, ids, counts, hist = train_rbk(X, C0, cfg)
    print(
        f"done: SSE={hist[-1]['SSE']:.4e} cv={hist[-1]['cv']:.4f} "
        f"max={counts.max()} min={counts.min()} empty={(counts==0).sum()}"
    )


if __name__ == "__main__":
    main()

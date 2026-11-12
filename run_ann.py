#!/usr/bin/env python3
"""Train RBK on an ANN base set (.fvecs / .npy / dataset directory)."""
from __future__ import annotations

import argparse

import faiss
import numpy as np

from rbk import RBKConfig, train_rbk
from rbk.io import load_base


def init_centroids(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k-means++ seed via Faiss (niter=0)."""
    d = X.shape[1]
    km = faiss.Kmeans(d, k, niter=0, nredo=1, seed=seed, verbose=False)
    km.train(np.ascontiguousarray(X.astype(np.float32)))
    return np.ascontiguousarray(km.centroids.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description="RBK on ANN datasets")
    ap.add_argument(
        "--data",
        required=True,
        help="path to base.fvecs / base.npy, or a dataset dir containing <name>_base.fvecs|.npy",
    )
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta0", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    faiss.omp_set_num_threads(int(args.threads))
    X = load_base(args.data)
    print(f"X shape={X.shape}", flush=True)

    C0 = init_centroids(X, args.k, seed=args.seed)
    cfg = RBKConfig(
        K=args.k,
        L=min(args.L, args.k),
        alpha=args.alpha,
        n_iters=args.iters,
        seed=args.seed,
        eta0=args.eta0,
        beta=args.beta,
        do_swap=not args.no_swap,
        do_post_merge=not args.no_merge,
    )
    C, ids, counts, hist = train_rbk(X, C0, cfg)
    for row in hist:
        print(
            f"iter={row['iter']:02d} SSE={row['SSE']:.6e} "
            f"cv={row['cv']:.4f} eta={row['eta']:.3f} "
            f"max={row['max_n']} min={row['min_n']}",
            flush=True,
        )
    print(
        f"done empty={(counts == 0).sum()} "
        f"cv={counts.astype(np.float64).std() / max(counts.mean(), 1e-12):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

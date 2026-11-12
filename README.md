# RBK

Regret-aware Balanced K-means for capacity-constrained IVF clustering.

RBK extends Lloyd iterations with approximate nearest-centroid search, capacity-aware assignment, light local refinement, and incremental centroid-graph maintenance, followed by a small post-cleanup before indexing.

## Install

```bash
pip install -r requirements.txt
make            # build C++ OpenMP kernels (g++ -O3/-Ofast -march=core-avx2)
```

Hot paths in `cpp/rbk_kernels.cpp` (capacity fill, centroid sum accumulation, SSE) are implemented in C++ with OpenMP and loaded from Python via ctypes. If the shared library is missing, pure-Python fallbacks are used.

## Hyperparameters (`RBKConfig` in `rbk/algorithm.py`)

```python
RBKConfig(
    K=4096,
    L=24,
    alpha=1.2,
    n_iters=12,
    hnsw_M=32,
    ef_construction=200,
    ef_search=64,
    drift_delta=1e-2,
    swap_top_frac=0.005,
    swap_top_cap=8000,
    swap_partners=16,
    eta0=1.0,
    beta=0.1,
    tiny_ratio=0.25,
)
```

Change defaults in `rbk/algorithm.py`, or override when calling `train_rbk`.

## Quick start (synthetic)

```bash
python examples/train_synthetic.py
```

```python
from rbk import RBKConfig, train_rbk
C, ids, counts, hist = train_rbk(X, C0, RBKConfig())
```

## Run on ANN datasets

Prepare a base set as `.fvecs`, `.npy`, or a directory with `<name>_base.fvecs` / `<name>_base.npy`.

```bash
python run_ann.py --data /path/to/sift --k 4096 --iters 12 --alpha 1.2 --L 24 --beta 0.1 --threads 8
python run_ann.py --data /path/to/gist/gist_base.fvecs --k 4096 --iters 12
```

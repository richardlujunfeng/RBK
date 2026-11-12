from __future__ import annotations

import os

import numpy as np


def read_fvecs(path: str) -> np.ndarray:
    fv = np.fromfile(path, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = int(fv.view(np.int32)[0])
    fv = fv.reshape(-1, 1 + dim)
    if not np.all(fv.view(np.int32)[:, 0] == dim):
        raise IOError(f"non-uniform vector sizes in {path}")
    return np.ascontiguousarray(fv[:, 1:].astype(np.float32))


def load_base(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.ascontiguousarray(np.load(path).astype(np.float32))
    if path.endswith(".fvecs"):
        return read_fvecs(path)
    if os.path.isdir(path):
        name = os.path.basename(os.path.normpath(path))
        npy = os.path.join(path, f"{name}_base.npy")
        fvecs = os.path.join(path, f"{name}_base.fvecs")
        if os.path.exists(npy):
            return np.ascontiguousarray(np.load(npy).astype(np.float32))
        if os.path.exists(fvecs):
            return read_fvecs(fvecs)
        raise FileNotFoundError(f"no base vectors under {path}")
    raise ValueError(f"unsupported path: {path}")

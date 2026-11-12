"""Legacy wrapper: prefer C++ kernels via rbk.native."""
from __future__ import annotations

import numpy as np

from .assign import fill_capacity as fill_capacity  # re-export
from .native import fill_capacity_cpp


def fill_capacity_py(order: np.ndarray, neigh: np.ndarray, rem: np.ndarray) -> np.ndarray:
    n = int(order.shape[0])
    L = int(neigh.shape[1])
    ids = np.full(n, -1, dtype=np.int32)
    rem = rem.astype(np.int32, copy=True)
    for t in range(n):
        i = int(order[t])
        row = neigh[i]
        for j in range(L):
            c = int(row[j])
            if c >= 0 and rem[c] > 0:
                ids[i] = c
                rem[c] -= 1
                break
    return ids

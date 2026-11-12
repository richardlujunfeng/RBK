from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Optional

import numpy as np

_LIB: Optional[ctypes.CDLL] = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(here, "fill_capacity.c")
    so = os.path.join(here, "fill_capacity.so")
    if (not os.path.exists(so)) or (os.path.getmtime(so) < os.path.getmtime(src)):
        subprocess.check_call(["gcc", "-O3", "-fPIC", "-shared", "-o", so, src])
    lib = ctypes.CDLL(so)
    lib.fill_capacity_i32.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.fill_capacity_i32.restype = None
    _LIB = lib
    return _LIB


def fill_capacity_py(order: np.ndarray, neigh: np.ndarray, rem: np.ndarray) -> np.ndarray:
    n = int(order.shape[0])
    L = int(neigh.shape[1])
    ids = np.full(n, -1, dtype=np.int32)
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


def fill_capacity(order: np.ndarray, neigh: np.ndarray, rem: np.ndarray) -> np.ndarray:
    order = np.ascontiguousarray(order, dtype=np.int64)
    neigh = np.ascontiguousarray(neigh, dtype=np.int32)
    rem = np.ascontiguousarray(rem, dtype=np.int32)
    n = int(order.shape[0])
    L = int(neigh.shape[1])
    ids = np.full(n, -1, dtype=np.int32)
    try:
        lib = _load_lib()
        lib.fill_capacity_i32(
            order.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            neigh.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            rem.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(n),
            ctypes.c_int64(L),
        )
        return ids
    except Exception:
        return fill_capacity_py(order, neigh, rem)

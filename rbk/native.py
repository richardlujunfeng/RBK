"""ctypes bindings to C++ OpenMP kernels (librbk_kernels.so)."""
from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Optional

import numpy as np

_LIB: Optional[ctypes.CDLL] = None


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_kernels(force: bool = False) -> str:
    """Compile librbk_kernels.so with g++ (-O3/-Ofast -march=core-avx2 -fopenmp)."""
    root = _repo_root()
    so = os.path.join(root, "librbk_kernels.so")
    src = os.path.join(root, "cpp", "rbk_kernels.cpp")
    hdr = os.path.join(root, "cpp", "rbk_kernels.h")
    if (
        force
        or (not os.path.exists(so))
        or os.path.getmtime(so) < os.path.getmtime(src)
        or os.path.getmtime(so) < os.path.getmtime(hdr)
    ):
        subprocess.check_call(["make", "-C", root, "all"])
    return so


def load_lib(force_rebuild: bool = False) -> Optional[ctypes.CDLL]:
    global _LIB
    if _LIB is not None and not force_rebuild:
        return _LIB
    try:
        so = build_kernels(force=force_rebuild)
        lib = ctypes.CDLL(so)
        lib.rbk_fill_capacity_i32.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        lib.rbk_fill_capacity_i32.restype = None
        lib.rbk_accumulate_sums_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        lib.rbk_accumulate_sums_f32.restype = None
        lib.rbk_sum_sq_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        lib.rbk_sum_sq_f32.restype = ctypes.c_double
        _LIB = lib
        return _LIB
    except Exception:
        _LIB = None
        return None


def fill_capacity_cpp(
    order: np.ndarray, neigh: np.ndarray, rem: np.ndarray
) -> Optional[np.ndarray]:
    lib = load_lib()
    if lib is None:
        return None
    order = np.ascontiguousarray(order, dtype=np.int64)
    neigh = np.ascontiguousarray(neigh, dtype=np.int32)
    rem = np.ascontiguousarray(rem, dtype=np.int32)
    n = int(order.shape[0])
    L = int(neigh.shape[1])
    ids = np.full(n, -1, dtype=np.int32)
    lib.rbk_fill_capacity_i32(
        order.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        neigh.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        rem.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(n),
        ctypes.c_int64(L),
    )
    return ids


def accumulate_sums_cpp(X: np.ndarray, ids: np.ndarray, K: int) -> Optional[np.ndarray]:
    lib = load_lib()
    if lib is None:
        return None
    X = np.ascontiguousarray(X, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    n, d = X.shape
    sums = np.zeros((K, d), dtype=np.float64)
    lib.rbk_accumulate_sums_f32(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        sums.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        ctypes.c_int64(d),
        ctypes.c_int64(K),
    )
    return sums


def sum_sq_cpp(X: np.ndarray, C: np.ndarray, ids: np.ndarray) -> Optional[float]:
    lib = load_lib()
    if lib is None:
        return None
    X = np.ascontiguousarray(X, dtype=np.float32)
    C = np.ascontiguousarray(C, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    n, d = X.shape
    return float(
        lib.rbk_sum_sq_f32(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(n),
            ctypes.c_int64(d),
        )
    )

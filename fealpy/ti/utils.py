from typing import Union

import numpy as np
import taichi as ti

dtype_map = {
    np.dtype(np.float32): ti.f32,
    np.dtype(np.float64): ti.f64,
    np.dtype(np.int32): ti.i32,
    np.dtype(np.int64): ti.i64,
    np.dtype(np.uint32): ti.u32,
    np.dtype(np.uint64): ti.u64,
}

# 定义一个映射函数来将 NumPy 的 dtype 转换为 Taichi 的 dtype
def numpy_to_taichi_dtype(np_dtype):
    if np_dtype in dtype_map:
        return dtype_map[np_dtype]
    else:
        raise ValueError(f"Unsupported dtype: {np_dtype}")

def to_taichi_field(arr):
    if isinstance(arr, np.ndarray):
        tarr = ti.field(dtype=dtype_map[arr.dtype], shape=arr.shape)
        tarr.from_numpy(arr)
        return tarr
    else:
        return arr

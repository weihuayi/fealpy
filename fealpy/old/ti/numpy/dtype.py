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

def to_ti_dtype(np_dtype):
    if np_dtype in dtype_map:
        return dtype_map[np_dtype]
    else:
        raise ValueError(f"Unsupported dtype: {np_dtype}")

from typing import Union, Optional

import numpy as np
import taichi as ti

# dtype map from numpy to taichi
dtype_map = {
    np.dtype(np.float32): ti.f32,
    np.dtype(np.float64): ti.f64,
    np.dtype(np.int32): ti.i32,
    np.dtype(np.int64): ti.i64,
    np.dtype(np.uint32): ti.u32,
    np.dtype(np.uint64): ti.u64,
}

def from_numpy(a: np.ndarray):
    """
    Create a Taichi field from a numpy.ndarray object.

    Parameters:
        a (np.ndarray): a numpy.ndarray
    """
    if isinstance(a, np.ndarray):
        tarr = ti.field(dtype=dtype_map[a.dtype], shape=a.shape)
        tarr.from_numpy(a)
        return tarr
    else:
        return a

def zeros(shape, dtype=ti.i32):
    """
    Create a Taichi field filled with zeros.

    Parameters:
        shape (tuple): Shape of the field.
        dtype (taichi.DataType): Data type of the field. Default is ti.i32.

    Returns:
        field: Taichi field filled with zeros.
    """
    field = ti.field(dtype, shape=shape)
    field.fill(0)
    return field

def ones(shape, dtype=ti.i32):
    """
    Create a Taichi field filled with ones.

    Parameters:
        shape (tuple): Shape of the field.
        dtype (taichi.DataType): Data type of the field. Default is ti.i32.

    Returns:
        field: Taichi field filled with ones.
    """
    field = ti.field(dtype, shape=shape)
    field.fill(1)
    return field

def arange(start: Union[int, float], 
           stop: Optional[Union[int, float]] = None, 
           step: Union[int, float] = 1, 
           dtype=ti.i32):
    """
    Create a Taichi field with evenly spaced values within a given interval.

    Parameters:
        start (int or float): Start of the interval.
        stop (int or float, optional): End of the interval. If not provided, start is treated as 0 and start is used as stop.
        step (int or float, optional): Spacing between values. Default is 1.
        dtype (taichi.DataType): Data type of the field. Default is ti.i32.

    Returns:
        field: Taichi field with evenly spaced values.
    """
    if stop is None:
        start, stop = 0, start

    d = stop - start
    num_elements = d//step + d%step 
    field = ti.field(dtype, shape=(num_elements,))

    @ti.kernel
    def fill_arange():
        for i in range(num_elements):
            field[i] = start + i * step

    fill_arange()
    return field


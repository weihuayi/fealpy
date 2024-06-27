from typing import Any, Union, Optional, TypeVar

import numpy as np
import taichi as ti

Field = TypeVar('Field')

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

def field(input_array: Any, dtype = None) -> Field:
    """
    Create a Taichi field from a given array-like input.

    Parameters:
        input_array (Any): The input data to be converted to a Taichi field. Can 
            be list, tuple, or other array-like object.
        dtype (optional): The data type of the Taichi field. If None, the data 
            type of the input array is used.

    Returns:
        ti.field: A Taichi field containing the input data.
    """
    input_array = np.asarray(input_array)
    shape = input_array.shape
    if dtype is None:
        dtype = dtype_map[input_array.dtype]
    ti_field = ti.field(dtype=dtype, shape=shape)
    ti_field.from_numpy(input_array)
    return ti_field

array = field


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


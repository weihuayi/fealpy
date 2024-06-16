from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import taichi as ti

Field = TypeVar('Field')

def sort(a: Field, axis: Optional[int]=-1, kind: str='quichsort', order=None):
    """
    Return a sorted copy of an array.

    Parameters:
        a (): field to be sorted.
        axis (int | None): Axis along which to sort. 
            If None, the array is flattened before sorting. The default is -1, which
            sorts along the last axis.
        kind (str): {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
            and 'mergesort' use timsort or radix sort under the covers and, in general,
            the actual implementation will vary with data type. The 'mergesort' option
            is retained for backwards compatibility.

    Returns:
        out (): Array of the same type and shape as `a`.

    See also:

    Notes:

    Examples:

    """
    M = a.shape[0]
    N = a.shape[1]
    out = ti.field(a.dtype, shape=a.shape)

    @ti.func
    def swap_row_elements(row, i, j):
        out[row, i], out[row, j] = out[row, j], out[row, i]

    @ti.func
    def partition(row, low, high):
        pivot = out[row, high]
        i = low - 1
        for j in range(low, high):
            if out[row, j] < pivot:
                i += 1
                swap_row_elements(row, i, j)
        swap_row_elements(row, i + 1, high)
        return i + 1

    @ti.func
    def quicksort(row, low, high):
        if low < high:
            pi = partition(row, low, high)
            quicksort(row, low, pi - 1)
            quicksort(row, pi + 1, high)

    @ti.kernel
    def sort_last_axis():
        for i in range(M):
            quicksort(i, 0, N - 1)

    # Initialize out with a
    @ti.kernel
    def copy_field():
        for i, j in ti.ndrange(M, N):
            out[i, j] = a[i, j]

    copy_field()
    if axis == -1 or axis == 1:
        sort_last_axis()
    else:
        raise NotImplementedError("Currently only sorting along the last axis is implemented.")

    return out
    

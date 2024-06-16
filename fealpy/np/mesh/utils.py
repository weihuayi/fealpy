from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


EntityName = Literal['cell', 'face', 'edge', 'node']
Entity = np.ndarray 
Index = Union[np.ndarray, int, slice]

_int_func = Callable[..., int]
_S = slice(None, None, None)
_T = TypeVar('_T')

def estr2dim(ds, estr: str) -> int:
    """
    Convert an entity string to its corresponding dimension.

    Parameters:
        ds: The mesh data structure which has a method `top_dimension()`.
        estr (str): Entity string representing the entity type. 
                    Should be one of 'cell', 'face', 'edge', or 'node'.

    Returns:
        int: The dimension of the entity.
             - 'cell' returns the top dimension of the mesh data structure.
             - 'face' returns the top dimension minus one.
             - 'edge' returns 1.
             - 'node' returns 0.

    Raises:
        KeyError: If the entity string `estr` is not one of the recognized entity attributes.
    """
    if estr == 'cell':
        return ds.top_dimension()
    elif estr == 'face':
        return ds.top_dimension() - 1
    elif estr == 'edge':
        return 1
    elif estr == 'node':
        return 0
    else:
        raise KeyError(f'{estr} is not a valid entity attribute.')

def ranges(nv, start = 0):
    shifts = np.cumsum(nv)
    id_arr = np.ones(shifts[-1], dtype=np.int_)
    id_arr[shifts[:-1]] = -np.asarray(nv[:-1])+1
    id_arr[0] = start
    return id_arr.cumsum()

def enable_csr(fn: Callable[..., NDArray]):
    """
    Decorator to make a function generating neighborhood information matrix
    support CSR (Compressed Sparse Row) matrix output.

    Parameters:
        fn (Callable[..., NDArray]): The function to be decorated.

    Returns:
        wrapped (Callable[..., Union[NDArray, csr_matrix]]): The wrapped function
        which can return either a dense ndarray or a sparse CSR matrix based
        on the `return_sparse` parameter.
    """
    @overload
    def wrapped(self, *args, return_sparse: Literal[False], **kwargs) -> NDArray: ...

    @overload
    def wrapped(self, *args, return_sparse: Literal[True], **kwargs) -> csr_matrix: ...

    def wrapped(self, *args, return_sparse=False, **kwargs):
        ret = fn(self, *args, **kwargs)
        if return_sparse:
            return arr_to_csr(arr=ret, reversed=False)
        else:
            return ret
    return wrapped


def arr_to_csr(arr: NDArray, n_col: Optional[int]=None,
               reversed=False, return_local=False, dtype=np.bool_):
    """
    Convert neighbor information matrix to sparse type.

    Parameters:
        arr (NDArray): The input ndarray representing the neighbor information matrix.
        n_col (int, optional): Number of columns of the output sparse matrix.
           For example, when converting `cell_to_edge` to sparse,
           `n_col` should be the number of edges, which will be
           the number of columns of the output sparse matrix.
           If not provided, `max(arr) + 1` is used.
        reversed (bool, optional): If True, transpose the sparse matrix, reversing the
                                   relationship. Default is False.
        return_local (bool, optional): Additional parameter to control the output.
                                       Default is False.
        dtype (np.dtype, optional): The data type of the output values. Default
        is np.bool_.

    Returns:
        csr_matrix: The converted sparse CSR matrix.

    Raises:
        ValueError: If the input array does not have 2 dimensions.
    """
    if arr.ndim != 2:
        raise ValueError("Can only tackle array with 2 dimensions.")
    nr, nv = arr.shape

    if n_col is not None:
        nc = n_col
    else:
        nc = np.max(arr) + 1

    if not return_local:
        val = np.ones(nr*nv, dtype=np.bool_)
    else:
        val = ranges(nv*np.ones(nr, dtype=dtype), start=1)

    if not reversed:
        sp = csr_matrix(
            (
                val,
                (
                    np.repeat(range(nr), nv),
                    arr.flat
                )
            ),
            shape=(nr, nc)
        )
        return sp
    else:
        sp = csr_matrix(
            (
                val,
                (
                    arr.flat,
                    np.repeat(range(nr), nv)
                )
            ),
            shape=(nc, nr)
        )
        return sp


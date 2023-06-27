from typing import Callable, overload, Literal, Optional
from scipy.sparse import csr_matrix
from numpy.typing import NDArray
import numpy as np

from ...common import ranges


def enable_csr(fn: Callable[..., NDArray]):
    """
    @brief Make a function generating neighborhood infomation matrix\
           supporting csr matrix output.
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
               reversed=False, return_local=False, dtype=np.int_):
    """
    @brief Convert neighbor information matrix to sparse type.

    @param arr: NDArray.
    @param n_col: int, optional. Number of columns of the output. For example,\
                  when converting `cell_to_edge` to sparse, `n_col` should be the\
                  number of edges, alse will be the number of columns of the output\
                  sparse matrix. If not provided, use `max(arr) + 1` instead.
    @param reversed: bool, defaults to False. Transpose the sparse matrix if True,\
                     then the relationship will be reversed.
    @oaram return_local: bool, deaults to False.
    @param dtype: np.dtype.
    """
    if arr.ndim != 2:
        raise ValueError("Can only tackle tensors with 2 dimension.")
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

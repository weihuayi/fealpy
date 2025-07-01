from typing import Callable, overload, Literal, Optional

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..common import ranges
from ..sparse import csr_matrix

def enable_csr(fn: Callable[..., TensorLike]):
    """
    @brief Make a function generating neighborhood infomation matrix\
           supporting csr matrix output.
    """
    @overload
    def wrapped(self, *args, return_sparse: Literal[False], **kwargs) -> TensorLike: ...
    @overload
    def wrapped(self, *args, return_sparse: Literal[True], **kwargs) -> csr_matrix: ...
    def wrapped(self, *args, return_sparse=False, **kwargs):
        ret = fn(self, *args, **kwargs)
        if return_sparse:
            return arr_to_csr(arr=ret, reversed=False)
        else:
            return ret

    return wrapped


def arr_to_csr(arr: TensorLike, n_col: Optional[int]=None,
               reversed=False, return_local=False, dtype=bm.int32):
    """
    @brief Convert neighbor information matrix to sparse type.

    @param arr: TensorLike.
    @param n_col: int, optional. Number of columns of the output. For example,\
                  when converting `cell_to_edge` to sparse, `n_col` should be the\
                  number of edges, alse will be the number of columns of the output\
                  sparse matrix. If not provided, use `max(arr) + 1` instead.
    @param reversed: bool, defaults to False. Transpose the sparse matrix if True,\
                     then the relationship will be reversed.
    @oaram return_local: bool, deaults to False.
    @param dtype: bm.int32.
    """
    if arr.ndim != 2:
        raise ValueError("Can only tackle tensors with 2 dimension.")
    nr, nv = arr.shape

    if n_col is not None:
        nc = n_col
    else:
        nc = bm.max(arr) + 1

    kwargs = bm.context(arr)
    if not return_local:
        val = bm.ones(nr*nv, **kwargs)
    else:
        val = ranges(nv*bm.ones(nr, **kwargs), start=1)

    if not reversed:
        sp = csr_matrix(
            (
                val,
                (
                    bm.repeat(bm.arange(nr), nv),
                    arr.reshape(-1)
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
                    arr.reshape(-1),
                    bm.repeat(bm.arange(nr), nv)
                )
            ),
            shape=(nc, nr)
        )
        return sp

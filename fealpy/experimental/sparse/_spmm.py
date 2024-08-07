
from typing import Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike as _DT

_Size = Tuple[int, ...]


def _shape_check(spshape: _Size, xshape: _Size):
    if len(spshape) != 2:
        raise ValueError("Sparse tensor to matrix multiplication must be"
                        f"2-D in sparse dims, but got {spshape}")

    if len(xshape) == 1:
        if spshape[-1] != xshape[0]:
            raise ValueError("Incompatible shapes detected in "
                             "sparse-dense matrix-vector multiplication, "
                            f"{spshape}(sparse dims) and {xshape}")
    elif len(xshape) >= 2:
        if spshape[-1] != xshape[-2]:
            raise ValueError("Incompatible shapes detected in "
                             "sparse-dense matrix multiplication, "
                            f"{spshape}(sparse dims) and {xshape}")
    else:
        raise ValueError(f"Illegal vector shape {xshape} found in the "
                         "sparse-dense multiplication")


def spmm_coo(indices: _DT, values: _DT, spshape: _Size, x: _DT) -> _DT:
    _shape_check(spshape, x.shape)
    row = indices[0]
    col = indices[1]

    if x.ndim == 1:
        new_vals = values * x[col]
        shape = new_vals.shape[:-1] + (spshape[0], )
        result = bm.zeros(shape, dtype=x.dtype)
        bm.index_add_(result, -1, row, new_vals)
        return result

    else: # x.ndim >= 2
        new_vals = values[..., None] * x[..., col, :] # (*batch, nnz, x_col)
        shape = new_vals.shape[:-2] + (spshape[0], x.shape[-1])
        result = bm.zeros(shape, dtype=x.dtype)
        bm.index_add_(result, -2, row, new_vals)
        return result


def spmm_csr(crow: _DT, col: _DT, values: _DT, spshape: _Size, x: _DT) -> _DT:
    _shape_check(spshape, x.shape)

    if x.ndim == 1:
        pass

    else: # x.ndim >= 2
        pass


from typing import Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike as _DT

_Size = Tuple[int, ...]


def spmm_coo(indices: _DT, values: _DT, spshape: _Size, x: _DT):
    if len(spshape) != 2:
        raise ValueError("COO tensor to matrix-vector multiplication must be"
                         f"2-D in sparse dims, but got shape {spshape}")

    row = indices[0]
    col = indices[1]

    if x.ndim == 1:
        if spshape[-1] != x.shape[0]:
            raise ValueError("COO tensor to matrix-vector multiplication must be"
                            f"compatible with vector, but got shape {spshape} and {x.shape}")

        new_vals = values * x[col]
        shape = new_vals.shape[:-1] + (spshape[0], )
        result = bm.zeros(shape, dtype=x.dtype)
        result = bm.index_add_(result, -1, row, new_vals)
        return result

    elif x.ndim >= 2:
        if spshape[-1] != x.shape[-2]:
            raise ValueError("COO tensor to matrix-vector multiplication must be"
                            f"compatible with vector, but got shape {spshape} and {x.shape}")

        new_vals = values[..., None] * x[..., col, :] # (*batch, nnz, x_col)
        shape = new_vals.shape[:-2] + (spshape[0], x.shape[-1])
        result = bm.zeros(shape, dtype=x.dtype)
        result = bm.index_add_(result, -2, row, new_vals)
        return result

    else:
        raise ValueError("COO tensor to matrix-vector multiplication only "
                         "supports 1D or 2D vectors, but got shape {}".format(x.shape))

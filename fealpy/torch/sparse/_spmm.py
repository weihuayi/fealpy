
from typing import Tuple

import torch
from torch import Tensor

_Size = Tuple[int, ...]


def spmm_coo(indices: Tensor, values: Tensor, spshape: _Size, x: Tensor):
    if len(spshape) != 2:
        raise ValueError("COO tensor to matrix-vector multiplication must be"
                         f"2-D in sparse dims, but got shape {spshape}")

    row = indices[0]
    col = indices[1]

    if x.dim() == 1:
        if spshape[-1] != x.shape[0]:
            raise ValueError("COO tensor to matrix-vector multiplication must be"
                            f"compatible with vector, but got shape {spshape} and {x.shape}")

        new_vals = values * x[col]
        shape = new_vals.shape[:-1] + (spshape[0], )
        result = torch.zeros(shape, dtype=x.dtype, device=x.device)
        dn = len(shape) - 1
        index = row[(None, )*dn + (slice(None), )].expand_as(new_vals)
        result = result.scatter_add_(-1, index, new_vals)
        return result

    elif x.dim() >= 2:
        if spshape[-1] != x.shape[-2]:
            raise ValueError("COO tensor to matrix-vector multiplication must be"
                            f"compatible with vector, but got shape {spshape} and {x.shape}")

        new_vals = values.unsqueeze(-1) * x[..., col, :] # (*batch, nnz, x_col)
        shape = new_vals.shape[:-2] + (spshape[0], x.shape[-1])
        result = torch.zeros(shape, dtype=x.dtype, device=x.device)
        dn = len(shape) - 2
        index = row[(None, )*dn + (slice(None), None)].expand_as(new_vals)
        result = result.scatter_add_(-2, index, new_vals)
        return result

    else:
        raise ValueError("COO tensor to matrix-vector multiplication only "
                         "supports 1D or 2D vectors, but got shape {}".format(x.shape))

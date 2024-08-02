
from typing import Tuple

import torch
from torch import Tensor

from ..utils import tril_coo


def ichol_coo(indices: Tensor, values: Tensor, size: Tuple[int, ...]) -> Tensor:
    """Incomplete cholesky factorization for sparse COO matrix.

    Parameters:
        indices (Tensor):
        values (Tensor):
        size (Size):

    Returns:
        Tensor: Lower triangular sparse COO matrix.
    """
    indices, values = tril_coo(indices, values)

    for col in range(size[0]):
        # positions of non-zeros in this column
        pos_this_col = (indices[1] == col)
        # row indices of non-zeros in this column
        row_nz_col = indices[0, :][pos_this_col]

        if col not in row_nz_col: # the diagonal element is zero
            continue

        pos_diag = (indices[0] == col) & pos_this_col
        values[pos_diag] = torch.sqrt(values[pos_diag])
        div = values[pos_diag]

        pos_lower_diag = (indices[0] > col) & pos_this_col
        values[pos_lower_diag] = values[pos_lower_diag] / div

        pos_greater_col = (indices[1] > col)

        for lower_row in indices[0][pos_lower_diag]:
            pos_this_lower_row = (indices[0] == lower_row)
            v1 = values[pos_this_lower_row & pos_this_col]
            pos_nz_this_row_greater_col = (indices[0] == lower_row) & pos_greater_col

            for greater_col in indices[1][pos_nz_this_row_greater_col]:
                pos_this_greater_col = (indices[1] == greater_col)
                v2 = values[(indices[0] == greater_col) & pos_this_col]

                if v1.numel() != 0 and v2.numel() != 0:
                    values[pos_this_lower_row & pos_this_greater_col] -= v1 * v2

    return indices, values

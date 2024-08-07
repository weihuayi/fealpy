
from typing import Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike as _DT

_Size = Tuple[int, ...]


def spspmm_coo(indices1: _DT, values1: _DT, spshape1: _Size,
               indices2: _DT, values2: _DT, spshape2: _Size) -> Tuple[_DT, _DT, _Size]:
    if len(spshape1) != 2 or len(spshape2) != 2:
        raise ValueError("COO tensors to matmul must be both 2-D for sparse dims, "
                         f"but got shape {spshape1} and {spshape2}")
    if spshape1[1] != spshape2[0]:
        raise ValueError(f"shapes must match: {spshape1} != {spshape2}")

    structure = values1.shape[:-1]
    if values2.shape[:-1] != structure:
        raise ValueError(f"the dense shape of matrix2 ({values2.shape[:-1]}) "
                         f"must match that of matrix1 {structure}")

    size = spshape1[1]
    indices_list = []
    values_list = []

    for i in range(size):
        left_col_flag = (indices1[1, :] == i)
        if not bm.any(left_col_flag):
            continue

        right_row_flag = (indices2[0, :] == i)
        if not bm.any(right_row_flag):
            continue

        row = indices1[0, left_col_flag]
        col = indices2[1, right_row_flag]
        nnz = col.shape[0] * row.shape[0]
        idx = bm.meshgrid(row, col, indexing='ij')
        idx = bm.reshape(bm.stack(idx, axis=0), (2, nnz))
        val1 = values1[..., left_col_flag]
        val2 = values2[..., right_row_flag]
        val = bm.einsum('...i, ...j -> ...ij', val1, val2).reshape(*structure, nnz)
        indices_list.append(idx)
        values_list.append(val)

    indices = bm.cat(indices_list, dim=1)
    values = bm.cat(values_list, dim=-1)

    return indices, values, (spshape1[0], spshape2[1])

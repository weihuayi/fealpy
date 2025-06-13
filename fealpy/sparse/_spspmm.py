
from typing import Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike as _DT

_Size = Tuple[int, ...]


def _shape_check(spshape1: _Size, spshape2: _Size):
    if len(spshape1) != 2 or len(spshape2) != 2:
        raise ValueError("Sparse tensors to matmul must be both 2-D for sparse dims, "
                        f"but got shape {spshape1} and {spshape2}")
    if spshape1[1] != spshape2[0]:
        raise ValueError("Incompatible shapes detected in "
                         "sparse-sparse matrix multiplication, "
                        f"got shape {spshape1} and {spshape2}.")


def spspmm_coo(indices1: _DT, values1: _DT, spshape1: _Size,
               indices2: _DT, values2: _DT, spshape2: _Size) -> Tuple[_DT, _DT, _Size]:
    _shape_check(spshape1, spshape2)

    structure = values1.shape[:-1]
    if values2.shape[:-1] != structure:
        raise ValueError(f"the dense shape of matrix2 ({values2.shape[:-1]}) "
                         f"must match that of matrix1 {structure}")

    size = spshape1[1]
    indices_list = []
    values_list = []

    for i in range(size):
        left_col_flag = (indices1[1, :] == i)

        if not bm.any(left_col_flag, axis=0):
            continue

        right_row_flag = (indices2[0, :] == i)

        if not bm.any(right_row_flag, axis=0):
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

    indices = bm.concat(indices_list, axis=1)
    values = bm.concat(values_list, axis=-1)
    return indices, values, (spshape1[0], spshape2[1])


def spspmm_csr(crow1: _DT, col1: _DT, values1: _DT, spshape1: _Size,
               crow2: _DT, col2: _DT, values2: _DT, spshape2: _Size) -> Tuple[_DT, _DT, _Size]:
    _shape_check(spshape1, spshape2)

    structure = values1.shape[:-1]
    if values2.shape[:-1] != structure:
        raise ValueError(f"the dense shape of matrix2 ({values2.shape[:-1]}) "
                         f"must match that of matrix1 {structure}")

    size = spshape1[0]
    indices_list = []
    values_list = []
    kargs = bm.context(crow1)
    new_crow = bm.zeros(size+1, **kargs)

    for i in range(size):
        row = bm.array([], **kargs)
        left_col_flag = (col1 == i)

        for x in bm.where(left_col_flag == True)[0]+1:
            flag = ~(x>crow1[:-1])^(x<=crow1[1:])
            row = bm.concat((row,(bm.where(flag == True))[0]))

        col = col2[crow2[i]:crow2[i+1]]

        if not bm.any(row, axis=0):
            continue

        if not bm.any(col, axis=0):
            continue

        idx = bm.meshgrid(row, col, indexing='ij')
        nnz = col.shape[0] * row.shape[0]
        idx = bm.reshape(bm.stack(idx, axis=0), (2, nnz))
        val1 = values1[..., left_col_flag]
        val2 = values2[..., crow2[i]:crow2[i+1]]
        val = bm.einsum('...i, ...j -> ...ij', val1, val2).reshape(*structure, nnz)

        indices_list.append(idx)
        values_list.append(val)

    indices = bm.concat(indices_list, axis=1)
    values = bm.concat(values_list, axis=-1)

    unique_indices, inverse_indices = bm.unique(indices, return_inverse=True, axis=1)
    new_values = bm.zeros(unique_indices.shape[1])
    new_values = bm.index_add(new_values, inverse_indices, values, axis=-1)

    for x in unique_indices[0,:]:
        # new_crow[x+1]+=1
        new_crow = bm.set_at(new_crow, x+1, new_crow[x+1])

    new_crow=bm.cumsum(new_crow, axis =-1)

    return new_crow, unique_indices[1,:], new_values,(spshape1[0], spshape2[1])


from typing import overload, Optional, Union, Literal

from ..backend import backend_manager as bm
from ..backend import TensorLike
from .coo_tensor import COOTensor
from .csr_tensor import CSRTensor


@overload
def spdiags(data: TensorLike, diags: Union[TensorLike, int], M: int, N: int) -> CSRTensor: ...
@overload
def spdiags(data: TensorLike, diags: Union[TensorLike, int], M: int, N: int,
            format: Literal['csr']) -> CSRTensor: ...
@overload
def spdiags(data: TensorLike, diags: Union[TensorLike, int], M: int, N: int,
            format: Literal['coo']) -> COOTensor: ...
def spdiags(data: TensorLike, diags: Union[TensorLike, int], M: int, N: int,
            format: Optional[str] = 'csr'):
    """Return a sparse matrix from diagonals.

    Parameters:
        data (Tensor): data on the matrix diagonals.
        diags (Tensor | int): index of matrix diagonals.
        for k in diags:
            k = 0 the main diagonal
            k > 0 the k-th upper diagonal
            k < 0 the k-th lower diagonal.

        M, N (int): shape of the result.
        format (str): format of the result, default to "csr".
    """
    is_scalar = False
    if data.ndim > 2:
        raise ValueError(f'the data must be a 2-D tensor, but got {data.ndim}-D')

    if isinstance(diags, TensorLike):
        diags = diags.flatten()
        if len(diags) > 1:
            if data.shape[0] != len(diags):
                raise ValueError(f'number of diagonals data: {data.shape[0]} does not match the number of diags: {len(diags)}')

            if len(bm.unique(diags)) != len(diags):
                raise ValueError('diags array contains duplicate values')

            diags = diags[:, None]
            num_diags, len_diags = data.shape

        else:
            is_scalar = True

    elif isinstance(diags, int):
        is_scalar = True
    else:
        raise TypeError(f"diags must be a tensor or int ,but got {type(diags)}")

    if is_scalar:
        if data.ndim == 1 or data.shape[0] == 1:
            data = data.flatten()
            num_diags = 1
            len_diags = data.shape[0]
        else:
            raise ValueError(f'number of diagonals data: {data.shape[0]} does not match the number of diags: 1')

    diags_inds = bm.arange(len_diags, device=data.device, dtype=bm.int64)
    row = diags_inds - diags

    mask = (row >= 0)
    mask &= (row < M)
    mask &= (diags_inds < N)
    mask &= (data != 0)
    row = row[mask]

    if is_scalar:
        col = diags_inds[mask]
    else:
        col = bm.tile(diags_inds, [num_diags])[mask.ravel()]

    data = data[mask]
    indices = bm.stack((row, col), axis=0)
    diag_tensor = COOTensor(indices, data, spshape=(M, N))

    if format == 'coo':
        return diag_tensor

    return diag_tensor.tocsr()

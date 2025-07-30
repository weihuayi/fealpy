
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

    diags_inds = bm.arange(len_diags, device=bm.get_device(data), dtype=bm.int64)
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

def vstack(blocks: TensorLike, format: Optional[str] = 'csr', dtype=None):
    if not isinstance(blocks, list) or not blocks: 
        raise ValueError('Blocks must be no empty.')

    if any(isinstance(item, list) for item in blocks):
        raise ValueError('Blocks must be 1-D')

    M = len(blocks)
    col = []
    values = []
    nb = 0
    nr = 0
    for i in range(M):
        if blocks[i] == None:
            continue
        if nb == 0:
            blocks_crow = [blocks[i].crow]
            nnz = blocks[i].nnz
            fblocks_idx = i

        nr = nr + blocks[i]._spshape[0]
        col.append(blocks[i].indices)
        values.append(blocks[i].values)
        if nb > 0:
            blocks_crow.append(nnz + blocks[i].crow[1:])
            nnz = nnz + blocks[i].nnz
        nb = nb + 1
    indices = bm.concat(col, axis=0)
    crow = bm.concat(blocks_crow, axis=0)
    values = bm.concat(values, axis=0)
    if dtype != None:
        values = values.astype(dtype)

    A = CSRTensor(crow, indices, values, spshape=(nr, blocks[fblocks_idx]._spshape[1]))
    if format == 'coo':
        return A.tocoo()
    return A

def hstack(blocks: TensorLike, format: Optional[str] = 'csr', dtype=None):
    if not isinstance(blocks, list) or not blocks: 
        raise ValueError('Blocks must be no empty.')

    if any(isinstance(item, list) for item in blocks):
        raise ValueError('Blocks must be 1-D')

    M = len(blocks)
    row_list = []
    col_list = []
    values_list = []
    cum_col = 0
    nb = 0
    for i in range(M):
        if blocks[i] == None:
            continue

        if nb == 0:
            fblocks_idx = i

        if nb > 0: 
            cum_col += blocks[i]._spshape[1]

        row_list.append(blocks[i].nonzero_slice[0])
        col_list.append(cum_col + blocks[i].nonzero_slice[1])
        values_list.append(blocks[i].values) 
        nb = nb + 1        
    row = bm.concat(row_list, axis=0)
    col = bm.concat(col_list, axis=0)
    indices = bm.stack((row, col), axis=0)
    values = bm.concat(values_list, axis=0)
    if dtype != None:
        values = values.astype(dtype)

    A = COOTensor(indices, values, spshape=(blocks[fblocks_idx]._spshape[0], cum_col + blocks[fblocks_idx]._spshape[1]))
    if format=='csr':
        return A.tocsr()
    return A


def bmat(blocks: TensorLike, format: Optional[str] = 'csr', dtype=None):
    if not isinstance(blocks, list) or not blocks: 
        raise ValueError('Blocks cannot be empty.')

    if not all(isinstance(item, list) for item in blocks):
        raise ValueError('Blocks must be 2-D')

    if any(isinstance(item2, list) for item1 in blocks for item2 in item1):
        raise ValueError('Blocks must be 2-D')

    M = len(blocks)
    N = len(blocks[0])

    if all(None not in blocks[b] for b in range(M)):
        if N > 1:
            blocks = [[hstack(blocks[b], format=format, dtype=dtype) for b in range(M)]]
        if M > 1:
            A = vstack(blocks[0], format=format, dtype=dtype)
        else:
            A = blocks[0]
        if dtype is not None:
            A = A.astype(dtype)
        return A

    ii = []
    jj = []
    nnz = 0
    for i in range(M):
        for j in range(N):
            if blocks[i][j] is not None:
                if nnz == 0:
                    kwargs1 = bm.context(blocks[i][j].crow)
                    kwargs2 = bm.context(blocks[i][j].values)
                    brow_lengths = bm.zeros(M, **kwargs1)
                    bcol_lengths = bm.zeros(N, **kwargs1)
                nnz = nnz + blocks[i][j].nnz

                A = blocks[i][j].tocoo()
                blocks[i][j] = A
                if brow_lengths[i] == 0:
                    brow_lengths[i] = A._spshape[0]
                elif brow_lengths[i] != A._spshape[0]:
                    msg = (f'blocks[{i},:] has incompatible row dimensions. '
                           f'Got blocks[{i},{j}].shape[0] == {A._spshape[0]}, '
                           f'expected {brow_lengths[i]}.')
                    raise ValueError(msg)
                ii.append(i)
                jj.append(j)
                if bcol_lengths[j] == 0:
                    bcol_lengths[j] = A._spshape[1]
                elif bcol_lengths[j] != A._spshape[1]:
                    msg = (f'blocks[:,{j}] has incompatible column '
                           f'dimensions. '
                           f'Got blocks[{i},{j}].shape[1] == {A._spshape[1]}, '
                           f'expected {bcol_lengths[j]}.')
                    raise ValueError(msg)

    row_offsets = bm.concat((bm.tensor([0], **kwargs1), bm.cumsum(brow_lengths, axis=0)))
    col_offsets = bm.concat((bm.tensor([0], **kwargs1), bm.cumsum(bcol_lengths, axis=0)))

    shape = (row_offsets[-1], col_offsets[-1])

    data = bm.empty(nnz, **kwargs2)
    row = bm.empty(nnz, **kwargs1)
    col = bm.empty(nnz, **kwargs1)

    nnz = 0
    for i, j in zip(ii, jj):
        B = blocks[i][j]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        row[idx] = bm.add(B.row, row_offsets[i])
        col[idx] = bm.add(B.col, col_offsets[j])
        nnz += B.nnz
    indices = bm.stack((row, col), axis=0)
    A = COOTensor(indices, data, spshape=shape)

    if format == 'csr':
        return A.tocsr()
    return A

@overload
def speye(M: int, N: Optional[int] = None, diags: Union[TensorLike, int] = 0, dtype=None,
          device = None) -> CSRTensor:...
@overload
def speye(M: int, N: Optional[int] = None, diags: Union[TensorLike, int] = 0, dtype=None,
          device = None, *, format: Literal['csr']) -> CSRTensor: ... 
@overload
def speye(M: int, N: Optional[int] = None, diags: Union[TensorLike, int] = 0, dtype=None,
          device = None, *, format: Literal['coo']) -> COOTensor: ... 
def speye(M: int, N: Optional[int] = None, diags: Union[TensorLike, int] = 0, dtype=None,
          device = None, *, format: Optional[str] = 'csr'):
    """Return a sparse matrix with ones on diagonal

    Parameters:  
        M (int): The number of rows of the resulting sparse tensor.
        N (int | None): The number of columns of the resulting sparse tensor.
        diags (Tensor | int): The index or indices of the diagonals to place ones on.
            k = 0: the main diagonal.
            k > 0: the k-th upper diagonal.
            k < 0: the k-th lower diagonal.

        dtype: The data type of the sparse tensor elements (the ones). Defaults to bm.int64.
        device: The device where the sparse tensor will be created (e.g., 'cpu', 'cuda').

        format (str | None): The format of the resulting sparse tensor.
    """
    if N is None:
        N = M

    if isinstance(diags, TensorLike):
        nd = len(diags)
    else:
        nd = 1

    values = bm.ones((nd, M), dtype=dtype, device=device)
    return spdiags(values, diags=diags, M=M, N=N, format=format)

def vstack(blocks: TensorLike, format: Optional[str] = 'csr', dtype=None):
    if not isinstance(blocks, list) or not blocks: 
        raise ValueError('Blocks must be no empty.')

    if any(isinstance(item, list) for item in blocks):
        raise ValueError('Blocks must be 1-D')

    M = len(blocks)
    col = []
    values = []
    nb = 0
    nr = 0
    for i in range(M):
        if blocks[i] == None:
            continue
        if nb == 0:
            blocks_crow = [blocks[i].crow]
            nnz = blocks[i].nnz
            fblocks_idx = i

        nr = nr + blocks[i]._spshape[0]
        col.append(blocks[i].indices)
        values.append(blocks[i].values)
        if nb > 0:
            blocks_crow.append(nnz + blocks[i].crow[1:])
            nnz = nnz + blocks[i].nnz
        nb = nb + 1
    indices = bm.concat(col, axis=0)
    crow = bm.concat(blocks_crow, axis=0)
    values = bm.concat(values, axis=0)
    if dtype != None:
        values = values.astype(dtype)

    A = CSRTensor(crow, indices, values, spshape=(nr, blocks[fblocks_idx]._spshape[1]))
    if format == 'coo':
        return A.tocoo()
    return A

def hstack(blocks: TensorLike, format: Optional[str] = 'csr', dtype=None):
    if not isinstance(blocks, list) or not blocks: 
        raise ValueError('Blocks must be no empty.')

    if any(isinstance(item, list) for item in blocks):
        raise ValueError('Blocks must be 1-D')

    M = len(blocks)
    row_list = []
    col_list = []
    values_list = []
    cum_col = 0
    nb = 0
    for i in range(M):
        if blocks[i] == None:
            continue

        if nb == 0:
            fblocks_idx = i

        if nb > 0: 
            cum_col += blocks[i]._spshape[1]

        row_list.append(blocks[i].nonzero_slice[0])
        col_list.append(cum_col + blocks[i].nonzero_slice[1])
        values_list.append(blocks[i].values) 
        nb = nb + 1        
    row = bm.concat(row_list, axis=0)
    col = bm.concat(col_list, axis=0)
    indices = bm.stack((row, col), axis=0)
    values = bm.concat(values_list, axis=0)
    if dtype != None:
        values = values.astype(dtype)

    A = COOTensor(indices, values, spshape=(blocks[fblocks_idx]._spshape[0], cum_col + blocks[fblocks_idx]._spshape[1]))
    if format=='csr':
        return A.tocsr()
    return A


def bmat(blocks: TensorLike, format: Optional[str] = 'csr', dtype=None):
    if not isinstance(blocks, list) or not blocks: 
        raise ValueError('Blocks cannot be empty.')

    if not all(isinstance(item, list) for item in blocks):
        raise ValueError('Blocks must be 2-D')

    if any(isinstance(item2, list) for item1 in blocks for item2 in item1):
        raise ValueError('Blocks must be 2-D')

    M = len(blocks)
    N = len(blocks[0])

    if all(None not in blocks[b] for b in range(M)):
        if N > 1:
            blocks = [[hstack(blocks[b], format=format, dtype=dtype) for b in range(M)]]
        if M > 1:
            A = vstack(blocks[0], format=format, dtype=dtype)
        else:
            A = blocks[0]
        if dtype is not None:
            A = A.astype(dtype)
        return A

    ii = []
    jj = []
    nnz = 0
    for i in range(M):
        for j in range(N):
            if blocks[i][j] is not None:
                if nnz == 0:
                    kwargs1 = bm.context(blocks[i][j].crow)
                    kwargs2 = bm.context(blocks[i][j].values)
                    brow_lengths = bm.zeros(M, **kwargs1)
                    bcol_lengths = bm.zeros(N, **kwargs1)
                nnz = nnz + blocks[i][j].nnz

                A = blocks[i][j].tocoo()
                blocks[i][j] = A
                if brow_lengths[i] == 0:
                    brow_lengths[i] = A._spshape[0]
                elif brow_lengths[i] != A._spshape[0]:
                    msg = (f'blocks[{i},:] has incompatible row dimensions. '
                           f'Got blocks[{i},{j}].shape[0] == {A._spshape[0]}, '
                           f'expected {brow_lengths[i]}.')
                    raise ValueError(msg)
                ii.append(i)
                jj.append(j)
                if bcol_lengths[j] == 0:
                    bcol_lengths[j] = A._spshape[1]
                elif bcol_lengths[j] != A._spshape[1]:
                    msg = (f'blocks[:,{j}] has incompatible column '
                           f'dimensions. '
                           f'Got blocks[{i},{j}].shape[1] == {A._spshape[1]}, '
                           f'expected {bcol_lengths[j]}.')
                    raise ValueError(msg)

    row_offsets = bm.concat((bm.tensor([0], **kwargs1), bm.cumsum(brow_lengths, axis=0)))
    col_offsets = bm.concat((bm.tensor([0], **kwargs1), bm.cumsum(bcol_lengths, axis=0)))

    shape = (row_offsets[-1], col_offsets[-1])

    data = bm.empty(nnz, **kwargs2)
    row = bm.empty(nnz, **kwargs1)
    col = bm.empty(nnz, **kwargs1)

    nnz = 0
    for i, j in zip(ii, jj):
        B = blocks[i][j]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        row[idx] = bm.add(B.row, row_offsets[i])
        col[idx] = bm.add(B.col, col_offsets[j])
        nnz += B.nnz
    indices = bm.stack((row, col), axis=0)
    A = COOTensor(indices, data, spshape=shape)

    if format == 'csr':
        return A.tocsr()
    return A
from ..sparse import COOTensor
from ..backend import backend_manager as bm

def spdiags(data, diags, M, N, format='csr'):
    if diags.ndim != 1:
        raise ValueError(f'diags must be a 1-D tensor, but got {diags.ndim}')

    if data.ndim != 2:
        raise ValueError(f'diags must be a 2-D tensor, but got {data.ndim}')

    if data.shape[0] != len(diags):
        raise ValueError(f'number of diagonals data: {data.shape[0]} does not match the number of offsets: {len(diags)}')

    if len(bm.unique(diags)) != len(diags):
        raise ValueError('diags array contains duplicate values')

    num_diags, len_diags = data.shape
    diags_inds = bm.arange(len_diags)
    row = diags_inds - diags[:, None]
    mask = (row >= 0)
    mask &= (row < M)
    mask &= (diags_inds < N)
    mask &= (data != 0)
    row = row[mask]
    col = bm.tile(diags_inds, [num_diags])[mask.ravel()]
    data = data[mask]

    indices = bm.stack((row, col), axis=0)
    diag_tensor = COOTensor(indices, data, spshape=(M, N))

    if format == 'coo':
        return diag_tensor

    return diag_tensor.tocsr()
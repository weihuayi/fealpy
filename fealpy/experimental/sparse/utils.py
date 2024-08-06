
from typing import Optional

from ..backend import backend_manager as bm
from ..backend import TensorLike, Size


def _dense_shape(values: Optional[TensorLike]):
    if values is None:
        return tuple()
    else:
        return values.shape[:-1]


def _dense_ndim(values: Optional[TensorLike]):
    if values is None:
        return 0
    else:
        return values.ndim - 1


def shape_to_strides(shape: Size, item_size: int):
    strides = [item_size, ]

    for i in range(1, len(shape)):
        strides.append(strides[-1] * shape[-i])

    return tuple(reversed(strides))


def _flatten_indices(indices: TensorLike, shape: Size) -> TensorLike:
    nnz = indices.shape[-1]
    strides = shape_to_strides(shape, 1)
    flatten = bm.zeros((nnz,), dtype=indices.dtype)

    for d, s in enumerate(strides):
        flatten += indices[d, :] * s

    return flatten[None, ...]


def check_shape_match(shape1: Size, shape2: Size):
    if shape1 != shape2:
        raise ValueError(f"shape mismatch: {shape1} != {shape2}")


def check_spshape_match(spshape1: Size, spshape2: Size):
    if spshape1 != spshape2:
        raise ValueError(f"sparse shape mismatch: {spshape1} != {spshape2}")


def tril_coo(indices: TensorLike, values: TensorLike, k: int=0) -> TensorLike:
    """Return the lower triangular portion (copy) of a sparse COO matrix.

    Parameters:
        indices (Tensor): _description_
        values (Tensor): _description_
        k (int, optional): The top-most diagonal of the lower triangle. Defaults to 0.

    Raises:
        ValueError: If the sparse matrix is not 2-dimensional.

    Returns:
        Tensor: new indices
        Tensor: new values
    """
    sparse_ndim = indices.shape[0]

    if sparse_ndim != 2:
        raise ValueError(f"indices must have 2 dimensions in tril, but got {sparse_ndim}")

    tril_pos = (indices[-2] + k) >= indices[-1]
    new_indices = bm.copy(indices[:, tril_pos])
    new_values = bm.copy(values[tril_pos])

    return new_indices, new_values

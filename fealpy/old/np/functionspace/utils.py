from typing import Optional, Tuple, Union

import numpy as np

from ..typing import Array
from ..typing import Size
from ..typing import _dtype


def zero_dofs(gdofs: int, dims: Union[Size, int, None]=None, *, dtype=None):
    if dims is None:
        shape = (gdofs,)
    elif isinstance(dims, int):
        shape = (gdofs,) if dims == 0 else (gdofs, dims)
    else:
        shape = (gdofs, *dims)

    return np.zeros(shape, dtype=dtype)


def flatten_indices(shape: Size, permute: Size) -> Array:
    """Construct indices of elements in the flattened tensor.

    Parameters:
        shape (Size): Shape of the source tensor.
        permute (Size): Permutation of the axes.

    Returns:
        Array: Indices of elements in the flattened tensor.
    """
    permuted_shape = tuple(shape[d] for d in permute)
    numel = np.prod(permuted_shape)
    permuted_indices = np.arange(numel).reshape(permuted_shape)
    inv_permute = [None] * len(permute)
    for d in range(len(permute)):
        inv_permute[permute[d]] = d
    return np.transpose(permuted_indices, inv_permute)


def to_tensor_dof(to_dof: Array, dof_numel: int, gdof: int, dof_priority: bool=True) -> Array:
    """Expand the relationship between entity and scalar dof to the tensor dof.

    Parameters:
        to_dof (Array): Entity to the scalar dof.
        dof_numel (int): Number of dof elements.
        gdof (int): Total number of dofs.
        dof_priority (bool, optional): If True, the degrees of freedom are arranged
        prior to their components. Defaults to True.

    Returns:
        Array: Global indices of tensor dofs in each entity.
    """
    num_entity = to_dof.shape[0]
    indices = np.arange(gdof * dof_numel, dtype=to_dof.dtype)

    if dof_priority:
        indices = indices.reshape(dof_numel, gdof).T
    else:
        indices = indices.reshape(gdof, dof_numel)

    return indices[to_dof].reshape(num_entity, -1)


def tensor_basis(shape: Tuple[int, ...], *, dtype: Optional[_dtype]=None) -> Array:
    """Generate tensor basis with 0-1 elements.

    Parameters:
        shape (Tuple[int, ...]): Shape of each tensor basis.

    Returns:
        Array: Tensor basis shaped (numel, *shape).
    """
    shape = tuple(shape)
    numel = np.prod(shape)
    return np.eye(numel, dtype=dtype).reshape((numel,) + shape)


def normal_strain(gphi: Array, indices: Array, *, out: Optional[Array]=None) -> Array:
    """Assembly normal strain tensor.

    Parameters:
        gphi (Array): Gradient of the scalar basis functions shaped (..., ldof, GD).
        indices (Array): Indices of DoF components in the flattened DoF, shaped (ldof, GD).
        out (Array | None, optional): Output tensor. Defaults to None.

    Returns:
        Array: Normal strain shaped (..., GD, GD*ldof).
    """
    ldof, GD = gphi.shape[-2:]
    new_shape = gphi.shape[:-2] + (GD, GD * ldof)

    if out is None:
        out = np.zeros(new_shape, dtype=gphi.dtype)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    for i in range(GD):
        out[..., i, indices[:, i]] = gphi[..., :, i]

    return out


def shear_strain(gphi: Array, indices: Array, *, out: Optional[Array]=None) -> Array:
    """Assembly shear strain tensor.

    Parameters:
        gphi (Array): Gradient of the scalar basis functions shaped (..., ldof, GD).
        indices (Array): Indices of DoF components in the flattened DoF, shaped (ldof, GD).
        out (Array | None, optional): Output tensor. Defaults to None.

    Returns:
        Array: Sheared strain shaped (..., NNZ, GD*ldof) where NNZ = (GD * (GD-1)) // 2.
    """
    ldof, GD = gphi.shape[-2:]
    if GD < 2:
        raise ValueError(f"The shear strain requires GD >= 2, but GD = {GD}")
    NNZ = (GD * (GD - 1)) // 2
    new_shape = gphi.shape[:-2] + (NNZ, GD * ldof)

    if out is None:
        out = np.zeros(new_shape, dtype=gphi.dtype)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    cursor = 0
    for i in range(0, GD - 1):
        for j in range(i + 1, GD):
            out[..., cursor, indices[:, i]] = gphi[..., :, j]
            out[..., cursor, indices[:, j]] = gphi[..., :, i]
            cursor += 1

    return out

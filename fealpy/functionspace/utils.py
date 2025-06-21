
from typing import Optional, Tuple, Union
from math import prod

from ..backend import backend_manager as bm
from ..typing import TensorLike, Size


def zero_dofs(gdofs: int, dims: Union[Size, int, None]=None, *, dtype=None):
    kwargs = {'dtype': dtype}

    if dims is None:
        shape = (gdofs, )
    elif isinstance(dims, int):
        shape = (gdofs, ) if dims == 0 else (gdofs, dims)
    else:
        shape = (gdofs, *dims)

    return bm.zeros(shape, **kwargs)


def flatten_indices(shape: Size, permute: Size) -> TensorLike:
    """Construct indices of elements in the flattened tensor.

    Parameters:
        shape (Tuple[int, ...]): Shape of the source tensor.
        permute (Tuple[int, ...]): _description_

    Returns:
        Tensor: Indices of elements in the flattened tensor.
    """
    permuted_shape = [shape[d] for d in permute]
    numel = prod(permuted_shape)
    # indices after permutation
    permuted_indices = bm.arange(numel, dtype=bm.int64).reshape(permuted_shape)
    # indices before permutation
    inv_permute = [None, ] * len(permute)
    for d in range(len(permute)):
        inv_permute[permute[d]] = d

    return bm.permute_dims(permuted_indices, inv_permute)


def to_tensor_dof(to_dof: TensorLike, dof_numel: int, gdof: int, dof_priority: bool=True) -> TensorLike:
    """Expand the relationship between entity and scalar dof to the tensor dof.

    Parameters:
        to_dof (Tensor): Entity to the scalar dofs.\n
        dof_numel (int): Number of dof elements.\n
        gdof (int): total number of scalar dofs.\n
        dof_priority (bool, optional): If True, the degrees of freedom are arranged\
        prior to their components. Defaults to True.

    Returns:
        Tensor: Global indices of tensor dofs in each entity.
    """
    context = bm.context(to_dof)
    indices = bm.arange(gdof*dof_numel, **context)
    num_entity = to_dof.shape[0]

    if dof_priority:
        indices = indices.reshape(dof_numel, gdof)
        indices = indices[:, to_dof] # (dof_numel, entity, ldof)
        indices = bm.swapaxes(indices, 0, 1) # (entity, dof_numel, ldof)
    else:
        indices = indices.reshape(gdof, dof_numel)
        indices = indices[to_dof, :] # (entity, ldof, dof_numel)

    return indices.reshape(num_entity, -1)


def tensor_basis(shape: Size, *, dtype=None, device=None) -> TensorLike:
    """Generate tensor basis with 0-1 elements.

    Parameters:
        shape (Tuple[int, ...]): Shape of each tensor basis.

    Returns:
        Tensor: Tensor basis shaped (numel, *shape).
    """
    numel = prod(shape)
    return bm.eye(numel, dtype=dtype, device=device).reshape((numel,) + shape)


def normal_strain(gphi: TensorLike, indices: TensorLike, *, out: Optional[TensorLike]=None) -> TensorLike:
    """Assembly normal strain tensor.

    Parameters:
        gphi (Tensor): Gradient of the scalar basis functions shaped (..., ldof, GD).\n
        indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (ldof, GD).\n
        out (Tensor | None, optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Normal strain shaped (..., GD, GD*ldof).
    """
    kwargs = {'dtype': gphi.dtype}
    if hasattr(gphi, 'device'):
        kwargs['device'] = gphi.device

    ldof, GD = gphi.shape[-2:]
    new_shape = gphi.shape[:-2] + (GD, GD*ldof)

    if out is None:
        out = bm.zeros(new_shape, **kwargs)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    for i in range(GD):
        out[..., i, indices[:, i]] = gphi[..., :, i]

    return out


def shear_strain(gphi: TensorLike, indices: TensorLike, *, out: Optional[TensorLike]=None) -> TensorLike:
    """Assembly shear strain tensor.

    Parameters:
        gphi (Tensor): Gradient of the scalar basis functions shaped (..., ldof, GD).\n
        indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (ldof, GD).\n
        out (Tensor | None, optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Sheared strain shaped (..., NNZ, GD*ldof) where NNZ = (GD + (GD+1))//2.
    """
    kwargs = {'dtype': gphi.dtype}
    if hasattr(gphi, 'device'):
        kwargs['device'] = gphi.device

    ldof, GD = gphi.shape[-2:]
    if GD < 2:
        raise ValueError(f"The shear strain requires GD >= 2, but GD = {GD}")
    NNZ = (GD * (GD-1))//2
    new_shape = gphi.shape[:-2] + (NNZ, GD*ldof)

    if out is None:
        out = bm.zeros(new_shape, **kwargs)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    cursor = 0
    for i in range(0, GD-1):
        for j in range(i+1, GD):
            out[..., cursor, indices[:, i]] = gphi[..., :, j]
            out[..., cursor, indices[:, j]] = gphi[..., :, i]
            cursor += 1

    return out

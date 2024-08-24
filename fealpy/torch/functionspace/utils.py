
from typing import Optional, Tuple, Union

import torch

from ..typing import Tensor
from ..typing import Size
from ..typing import _dtype, _device


def zero_dofs(gdofs: int, dims: Union[Size, int, None]=None, *, dtype=None, device=None):
    kwargs = {'device': device, 'dtype': dtype}

    if dims is None:
        shape = (gdofs, )
    elif isinstance(dims, int):
        shape = (gdofs, ) if dims == 0 else (gdofs, dims)
    else:
        shape = (gdofs, *dims)

    return torch.zeros(shape, **kwargs)


def flatten_indices(shape: Size, permute: Size) -> Tensor:
    """Construct indices of elements in the flattened tensor.

    Parameters:
        shape (_Size): Shape of the source tensor.
        permute (_Size): _description_

    Returns:
        Tensor: Indices of elements in the flattened tensor.
    """
    permuted_shape = torch.Size([shape[d] for d in permute])
    numel = permuted_shape.numel()
    permuted_indices = torch.arange(numel, dtype=torch.long).reshape(permuted_shape)
    inv_permute = [None, ] * len(permute)
    for d in range(len(permute)):
        inv_permute[permute[d]] = d
    return permuted_indices.permute(inv_permute)


def to_tensor_dof(to_dof: Tensor, dof_numel: int, gdof: int, dof_priority: bool=True) -> Tensor:
    """Expand the relationship between entity and scalar dof to the tensor dof.

    Parameters:
        to_dof (Tensor): Entity to the scalar dof.\n
        dof_numel (int): Number of dof elements.\n
        gdof (int): total number of dofs.\n
        dof_priority (bool, optional): If True, the degrees of freedom are arranged\
        prior to their components. Defaults to True.

    Returns:
        Tensor: Global indices of tensor dofs in each entity.
    """
    kwargs = {'dtype': to_dof.dtype, 'device': to_dof.device}
    num_entity = to_dof.shape[0]
    indices = torch.arange(gdof*dof_numel, **kwargs)

    if dof_priority:
        indices = indices.reshape(dof_numel, gdof).T
    else:
        indices = indices.reshape(gdof, dof_numel)

    return indices[to_dof].reshape(num_entity, -1)


def tensor_basis(shape: Tuple[int, ...], *, dtype: Optional[_dtype]=None,
                 device: Union[str, _device, None]=None) -> Tensor:
    """Generate tensor basis with 0-1 elements.

    Parameters:
        shape (Tuple[int, ...]): Shape of each tensor basis.

    Returns:
        Tensor: Tensor basis shaped (numel, *shape).
    """
    kwargs = {'dtype': dtype, 'device': device}
    shape = torch.Size(shape)
    numel = shape.numel()
    return torch.eye(numel, **kwargs).reshape((numel,) + shape)


def normal_strain(gphi: Tensor, indices: Tensor, *, out: Optional[Tensor]=None) -> Tensor:
    """Assembly normal strain tensor.

    Parameters:
        gphi (Tensor): Gradient of the scalar basis functions shaped (..., ldof, GD).\n
        indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (ldof, GD).\n
        out (Tensor | None, optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Normal strain shaped (..., GD, GD*ldof).
    """
    kwargs = {'dtype': gphi.dtype, 'device': gphi.device}
    ldof, GD = gphi.shape[-2:]
    new_shape = gphi.shape[:-2] + (GD, GD*ldof)

    if out is None:
        out = torch.zeros(new_shape, **kwargs)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    for i in range(GD):
        out[..., i, indices[:, i]] = gphi[..., :, i]

    return out


def shear_strain(gphi: Tensor, indices: Tensor, *, out: Optional[Tensor]=None) -> Tensor:
    """Assembly shear strain tensor.

    Parameters:
        gphi (Tensor): Gradient of the scalar basis functions shaped (..., ldof, GD).\n
        indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (ldof, GD).\n
        out (Tensor | None, optional): Output tensor. Defaults to None.

    Returns:
        Tensor: Sheared strain shaped (..., NNZ, GD*ldof) where NNZ = (GD + (GD+1))//2.
    """
    kwargs = {'dtype': gphi.dtype, 'device': gphi.device}
    ldof, GD = gphi.shape[-2:]
    if GD < 2:
        raise ValueError(f"The shear strain requires GD >= 2, but GD = {GD}")
    NNZ = (GD * (GD-1))//2
    new_shape = gphi.shape[:-2] + (NNZ, GD*ldof)

    if out is None:
        out = torch.zeros(new_shape, **kwargs)
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

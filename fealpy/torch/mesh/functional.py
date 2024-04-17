
from typing import Optional

import torch
from torch import Tensor

_dtype = torch.dtype


def homo_mesh_top_coo(entity: Tensor, num_targets: int, *,
                      dtype: Optional[_dtype]=None) -> Tensor:
    r"""COOrdinate format of a homogeneous mesh topology relaionship matrix."""
    kwargs = {'dtype': entity.dtype, 'device': entity.device}
    num = entity.numel()
    num_source = entity.size(0)
    indices = torch.zeros((2, num), **kwargs)
    indices[0, :] = torch.arange(num_source, **kwargs).repeat_interleave(entity.size(1))
    indices[1, :] = entity.reshape(-1)
    return torch.sparse_coo_tensor(
        indices,
        torch.ones(num, dtype=dtype, device=entity.device),
        size=(num_source, num_targets),
        dtype=dtype, device=entity.device
    )


def mesh_top_csr(entity: Tensor, num_targets: int, location: Optional[Tensor]=None, *,
                 dtype: Optional[_dtype]=None) -> Tensor:
    r"""CSR format of a mesh topology relaionship matrix."""
    device = entity.device

    if entity.ndim == 1: # for polygon case
        if location is None:
            raise ValueError('location is required for 1D entity (usually for polygon mesh).')
        crow = location
    elif entity.ndim == 2: # for homogeneous case
        crow = torch.arange(
            entity.size(0) + 1, dtype=entity.dtype, device=device
        ).mul_(entity.size(1))
    else:
        raise ValueError('dimension of entity must be 1 or 2.')

    return torch.sparse_csr_tensor(
        crow,
        entity.reshape(-1),
        torch.ones(entity.numel(), dtype=dtype, device=device),
        size=(entity.size(0), num_targets),
        dtype=dtype, device=device
    )

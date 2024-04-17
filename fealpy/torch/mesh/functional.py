
import torch
from torch import Tensor


def homo_mesh_top_coo(entity: Tensor, num_targets: int) -> Tensor:
    kwargs = {'dtype': entity.dtype, 'device': entity.device}
    num = entity.numel()
    num_source = entity.size(0)
    indices = torch.zeros((2, num), **kwargs)
    indices[0, :] = torch.arange(num_source, **kwargs).repeat_interleave(entity.size(1))
    indices[1, :] = entity.reshape(-1)
    return torch.sparse_coo_tensor(
        indices,
        torch.ones(num, dtype=torch.bool, device=entity.device),
        size=(num_source, num_targets),
        **kwargs
    )


def homo_mesh_top_csr(entity: Tensor, num_targets: int) -> Tensor:
    kwargs = {'dtype': entity.dtype, 'device': entity.device}
    crow = torch.arange(entity.size(0) + 1, **kwargs) * entity.size(1)
    return torch.sparse_csr_tensor(
        crow,
        entity.reshape(-1),
        torch.ones(entity.numel(), dtype=torch.bool, device=entity.device),
        size=(entity.size(0), num_targets),
        **kwargs
    )

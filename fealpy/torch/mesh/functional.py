
import torch
from torch import Tensor


def homo_mesh_top_coo_indices(entity: Tensor, num_targets: int):
    kwargs = {'dtype': entity.dtype, 'device': entity.device}
    num = entity.numel()
    num_source = entity.size(0)
    indices = torch.zeros((num, 2), **kwargs)
    indices[:, 0] = torch.arange(num_source, **kwargs).repeat_interleave(num_targets)
    indices[:, 1] = entity.reshape(-1)
    return indices

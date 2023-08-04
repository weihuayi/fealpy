
import torch
from torch import Tensor

from .nntyping import TensorFunction
from .sampler import Sampler


def integral(fn: TensorFunction, sampler: Sampler):
    x = sampler.run()
    val = fn(x)
    return torch.mean(val, dim=0)


def linf_error(fn: TensorFunction, target: TensorFunction, sampler: Sampler) -> Tensor:
    x = sampler.run()
    delta = fn(x) - target(x)
    return torch.norm(delta, p=torch.inf, dim=0)

def l2_error(fn: TensorFunction, target: TensorFunction, sampler: Sampler) -> Tensor:
    x = sampler.run()
    delta = fn(x) - target(x)
    return torch.norm(delta, p=2, dim=0)


import torch
from torch import Tensor

from .nntyping import TensorFunction
from .sampler import Sampler


def integral(fn: TensorFunction, sampler: Sampler):
    """
    @brief Integral by Monte Carlo Method.

    @note: Weights of samples are required to be implemented.
    """
    x = sampler.run()
    val = fn(x) * sampler.weight()
    return torch.mean(val, dim=0)


def linf_error(fn: TensorFunction, target: TensorFunction, sampler: Sampler) -> Tensor:
    """
    @brief Estimate L-infinity error between fn and target by Monte Carlo method.

    @note: Functions and the sampler should be in a same device.
    """
    x = sampler.run()
    delta = fn(x) - target(x)
    return torch.norm(delta, p=torch.inf, dim=0)


def l2_error(fn: TensorFunction, target: TensorFunction, sampler: Sampler) -> Tensor:
    """
    @brief Estimate L-2 error between fn and target by Monte Carlo method.
    """
    x = sampler.run()
    delta = fn(x) - target(x)
    w = sampler.weight()
    delta *= torch.sqrt(w)
    return torch.norm(delta, p=2, dim=0)


from typing import Sequence
import torch
from torch import Tensor, exp
from .pde import PDE


class Gaussian2d(PDE):
    def __init__(self, domain: Sequence[float]) -> None:
        super().__init__()
        self._domain = tuple(domain)

    def solution(self, p: Tensor):
        x = p[:, 0:1]
        y = p[:, 1:2]
        return exp(-0.5 * (x**2 + y**2))

    def source(self, p: Tensor):
        x = p[:, 0:1]
        y = p[:, 1:2]
        return -(x**2 + y**2 - 2) * exp(-0.5 * (x**2 + y**2))

    def gradient(self, p: Tensor):
        x = p[:, 0:1]
        y = p[:, 1:2]
        return torch.cat([
            -x * exp(-0.5 * (x**2 + y**2)),
            -y * exp(-0.5 * (x**2 + y**2))
        ], dim=-1)

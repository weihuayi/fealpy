
import torch
from torch.nn import Module
from torch import Tensor

from ..nntyping import TensorFunction


class GradAttention(Module):
    def __init__(self, grad_f: TensorFunction, alpha: float=0.1) -> None:
        super().__init__()
        self.df = grad_f
        self.alpha = alpha

    def forward(self, p: Tensor):
        assert p.ndim == 2

        g = self.df(p)
        dp = p[None, :, :] - p[:, None, :]
        weights = torch.inner(g, g)
        weights = torch.softmax(-weights, dim=-1)

        return p + torch.einsum('ijd, ij -> id', dp, weights)*self.alpha


from typing import Optional
import torch
from torch import Tensor
from torch.nn import functional as F
from .module import TensorMapping
torch.nn.MSELoss

class ScaledMSELoss(TensorMapping):
    """
    @brief Scale by the square of largest element (by abs) after the normal MSE.
    """
    __constants__ = ['scale']

    def __init__(self, scale: float=100.0, tol: float=1e-8) -> None:
        super().__init__()
        self.scale = scale
        self.tol = tol

    def forward(self, input: Tensor, target: Optional[Tensor]=None):
        if target is None:
            target = torch.zeros_like(input)
        raw = F.mse_loss(input, target, reduction='none')
        max_val = torch.max(raw).detach()
        if max_val < self.tol:
            max_val = torch.tensor(1.0)
        return self.scale/max_val * torch.mean(raw)

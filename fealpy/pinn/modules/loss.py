
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

    def __init__(self, scale: float=100.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, input: Tensor, target: Optional[Tensor]=None):
        if target is None:
            target = torch.zeros_like(input)
        raw = F.mse_loss(input, target, reduction='none')
        lambda_ = self.scale/torch.max(raw)
        return lambda_*torch.mean(raw)

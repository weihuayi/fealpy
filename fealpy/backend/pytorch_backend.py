
import torch

from .base import Backend

Tensor = torch.Tensor


class PyTorchBackend(Backend[Tensor], backend_name='pytorch'):
    DATA_CLASS = torch.Tensor

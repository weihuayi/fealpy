
import torch

from .space import FunctionSpace

_Size = torch.Size


class TensorFunctionSpace(FunctionSpace):
    def __init__(self, scalar_space: FunctionSpace, shape: _Size) -> None:
        self.scalar_space = scalar_space
        self.shape = shape

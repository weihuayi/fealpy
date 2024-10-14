"""
Modules for the Physics-Informed Kernel Function Neural Networks
"""

from torch import Tensor

from ..nntyping import TensorFunction, S
from .linear import Distance
from .function_space import FunctionSpace


class KernelFunctionSpace(FunctionSpace):
    def __init__(self, sources: Tensor, kernel: TensorFunction, device=None) -> None:
        """
        @brief
        """
        super().__init__()
        self.in_dim = sources.shape[-1]
        self.out_dim = 1
        self.ns = sources.shape[0]

        self.radius_layer = Distance(sources=sources, device=device)

        self.kernel = kernel

    @property
    def source(self):
        return self.radius_layer.sources

    def number_of_basis(self) -> int:
        return self.ns

    def basis(self, p: Tensor, *, index=S) -> Tensor:
        return self.kernel(self.radius_layer(p))

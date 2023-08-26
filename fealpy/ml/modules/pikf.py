"""
Modules for the Physics-Informed Kernel Function Neural Networks
"""

from torch import Tensor

from .linear import Distance
from .function_space import TensorSpace


class KernelFunctionSpace(TensorSpace):
    def __init__(self, sources: Tensor, kernel, device=None) -> None:
        """
        @brief
        """
        super().__init__()
        self.in_dim = sources.shape[-1]
        self.out_dim = 1
        self.ns = sources.shape[0]

        self.radius_layer = Distance(sources=sources, p=2, device=device)

        self.kernel = kernel

    def number_of_basis(self) -> int:
        return self.ns

    def basis_value(self, p: Tensor) -> Tensor:
        return self.kernel(self.radius_layer(p))

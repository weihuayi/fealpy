
from typing import Tuple
import torch
from torch import Tensor
from torch.func import jvp


class PDE():
    """
    Base class for all PDE problem settings.
    """
    _domain: Tuple[float, ...]
    def domain(self):
        """
        @brief Return the domain of the model.
        """
        return getattr(self, '_domain', None)

    def solution(self, p: Tensor) -> Tensor:
        """
        @brief Return the value of solution at `p`, with shape (..., 1).
        """
        raise NotImplementedError

    def source(self, p: Tensor) -> Tensor:
        """
        @brief Return the value of source term of the PDE, with shape (..., 1).
        """
        raise NotImplementedError

    def gradient(self, p: Tensor) -> Tensor:
        """
        @brief Return the gradient of solution at `p`, with shape (..., #dims).
        """
        raise NotImplementedError

    # boundary conditions
    def dirichlet(self, p: Tensor):
        """
        @brief Dirichlet boundary condition at `p`, with shape (..., 1).
        """
        return self.solution(p)

    def neumann(self, p: Tensor, n: Tensor):
        """
        @brief Neumann boundary condition at `p`, with shape (..., 1).
        """
        grad = self.gradient(p)
        return torch.sum(grad*n, dim=-1, keepdim=True)

    def robin(self, p: Tensor, n: Tensor, kappa: float):
        """
        @brief Robin boundary condition at `p`, with shape (..., 1).
        """
        grad = self.gradient(p)
        return torch.sum(grad*n, dim=-1, keepdim=True) + kappa * self.solution(p)

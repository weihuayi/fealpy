"""
Modules for the Random Feature Method
"""

from typing import Tuple

import torch
from torch import Tensor, float64
from torch.nn import init, Linear

from .function_space import FunctionSpaceBase
from .activate import Activation

PI = torch.pi


class RandomFeatureSpace(FunctionSpaceBase):
    def __init__(self, in_dim: int, nf: int,
                 activate: Activation,
                 bound: Tuple[float, float]=(1.0, PI),
                 dtype=float64, device=None) -> None:
        """
        @brief Construct a random feature model.

        @param in_dim: int. Dimension of inputs.
        @param nf: int. Number of random features.
        @param activate: Activation.
        @param bound: two floats. Bound of uniform distribution to initialize\
               k, b in the each random feature.
        @param dtype: torch.dtype. Data type of inputs.
        @param device: torch.device.
        """
        super().__init__(in_dim, 1, dtype, device)
        self.nf = nf

        self.linear = Linear(in_dim, nf, device=device, dtype=dtype)
        self.linear.requires_grad_(False)
        self._set_basis(bound)

        self.activate = activate

    def _set_basis(self, bound: Tuple[float, float]):
        init.uniform_(self.linear.weight, -bound[0], bound[0])
        init.uniform_(self.linear.bias, -bound[1], bound[1])

    @property
    def frequency(self):
        """
        @brief The "frequency" of the basis. Shape: (nf, GD).
        """
        return self.linear.weight

    @property
    def phrase(self):
        """
        @brief The "phrase" of the basis. Shape: (nf, ).
        """
        return self.linear.bias

    def number_of_basis(self):
        return self.nf

    def _weight_transform(self, transform=None):
        if transform is None:
            return self.linear.weight
        else:
            return torch.einsum("fu, ux -> fx", self.linear.weight, transform)

    def basis(self, p: Tensor) -> Tensor:
        """
        @brief Return values of basis, with shape (N, nf).
        """
        return self.activate(self.linear(p))

    def grad_basis(self, p: Tensor, transform=None) -> Tensor:
        """
        @brief Return gradient vector of basis, with shape (N, nf, GD).
        """
        a = self.activate.d1(self.linear(p))
        return torch.einsum("nf, fx -> nfx", a, self._weight_transform(transform))

    def hessian_basis(self, p: Tensor, transform=None) -> Tensor:
        """
        @brief Return hessian matrix of basis, with shape (N, nf, GD, GD).
        """
        a = self.activate.d2(self.linear(p))
        grad = self._weight_transform(transform)
        return torch.einsum("nf, fx, fy -> nfxy", a, grad, grad)

    def laplace_basis(self, p: Tensor, coef=None, transform=None) -> Tensor:
        """
        @brief Return basis evaluated by laplace operator, with shape (N, nf).
        """
        a = self.activate.d2(self.linear(p))
        grad = self._weight_transform(transform)
        if coef is None:
            return torch.einsum("nf, fd, fd -> nf", a, grad, grad)
        else:
            if coef.ndim == 1:
                return torch.einsum("nf, d, fd, fd -> nf", a, coef, grad, grad)
            else:
                return torch.einsum("nf, nd, fd, fd -> nf", a, coef, grad, grad)

    def derivative_basis(self, p: Tensor, *idx: int, transform=None) -> Tensor:
        """
        @brief Return specified partial derivatives of basis, with shape (N, nf).

        @param *idx: int. index of the independent variable to take partial derivatives.
        """
        order = len(idx)
        if order == 0:
            return self.activate(self.linear(p))
        else:
            grad = self._weight_transform(transform)
            a = self.activate.dn(self.linear(p), order)
            b = torch.prod(grad[:, idx], dim=-1, keepdim=False)
            return torch.einsum("nf, f -> nf", a, b)


#     def scale(self, p: Tensor, operator: Operator):
#         """
#         @brief Return the scale by basis and operator.
#         @note: This method may need autograd function of PyTorch in `operator`.\
#                If the operator object is not based on autograd, this method can\
#                not help to get a scale.
#         """
#         MP = self.number_of_partitions()
#         std = self.std(p)
#         partition_max = torch.zeros((MP, ), dtype=self.dtype, device=self.device)
#         for idx, part in enumerate(self.partions):
#             x = std[:, idx, :]
#             flag = part.flag(x) # Only take samples inside the supporting area
#             psiphi = self.partions[idx].basis(x[flag, ...]) # (N, nf)
#             partition_max[idx] = torch.max(operator(p, psiphi))
#         return torch.max(partition_max)

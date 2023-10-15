"""
Modules for the Random Feature Method
"""

from typing import Tuple

import torch
from torch import Tensor, float64
from torch.nn import init, Linear
from torch.nn import functional as F

from ..nntyping import S as _S
from .function_space import FunctionSpace
from .activate import Activation

PI = torch.pi


class RandomFeatureSpace(FunctionSpace):
    """
    The random feature space is a function space whose basis functions are\
    random features.
    """
    def __init__(self, in_dim: int, nf: int,
                 activate: Activation,
                 bound: Tuple[float, float]=(1.0, PI),
                 dtype=float64, device=None) -> None:
        """
        @brief Construct a random feature space.

        @param in_dim: int. The geometry dimension of inputs.
        @param nf: int. Number of random features (basis functions).
        @param activate: Activation. The activation function after the linear\
               transformation.
        @param bound: two floats. Bounds of uniform distribution to initialize\
               k, b in the each random feature.
        @param dtype: torch.dtype. Data type of inputs.
        @param device: torch.device.
        """
        super().__init__(in_dim=in_dim, out_dim=1, dtype=dtype, device=device)
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

    def _linear(self, p: Tensor, *, index=_S):
        return F.linear(p, self.linear.weight[index, :], self.linear.bias[index])

    def basis(self, p: Tensor, *, index=_S) -> Tensor:
        return self.activate(self._linear(p, index=index))

    def grad_basis(self, p: Tensor, *, index=_S) -> Tensor:
        a = self.activate.d1(self._linear(p, index=index))
        grad = self.linear.weight[index, :]
        return torch.einsum("...f, fx -> ...fx", a, grad)

    def hessian_basis(self, p: Tensor, *, index=_S) -> Tensor:
        a = self.activate.d2(self._linear(p, index=index))
        grad = self.linear.weight[index, :]
        return torch.einsum("...f, fx, fy -> ...fxy", a, grad, grad)

    def laplace_basis(self, p: Tensor, *, index=_S) -> Tensor:
        a = self.activate.d2(self._linear(p, index=index))
        grad = self.linear.weight[index, :]
        return torch.einsum("...f, fd, fd -> ...f", a, grad, grad)

    def derivative_basis(self, p: Tensor, *idx: int, index=_S) -> Tensor:
        order = len(idx)
        if order == 0:
            return self.activate(self._linear(p, index=index))
        else:
            a = self.activate.dn(self._linear(p, index=index), order)
            b = torch.prod(self.linear.weight[:, idx], dim=-1, keepdim=False)
            return torch.einsum("...f, f -> ...f", a, b)


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

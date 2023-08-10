"""Modules for the Random Feature Method"""

from typing import List

import torch
from torch import Tensor
from torch.nn import Module, Parameter, init, Linear
from torch.nn.parameter import Parameter

from ..nntyping import Operator
from .linear import MultiLinear, StackStd
from .module import TensorMapping


class PoU(Module):
    def __init__(self, keepdim=False) -> None:
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: Tensor): # (N, Mp, d)
        flag = (-1 <= x) * (x < 1)
        flag = torch.prod(flag, dim=-1, keepdim=self.keepdim)
        return flag.double()


class PoUSin(Module):
    """
    @brief Sin-style partition of unity.

    For inputs with shape (..., d), the output is like (..., ) or (..., 1),\
    and values of each element is between 0 and 1.
    """
    def __init__(self, keepdim=False) -> None:
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: Tensor): # (N, Mp, d) -> (N, Mp)
        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        l1 = 0.5 * (1 + torch.sin(2*torch.pi*x)) * f1
        l2 = f2.double()
        l3 = 0.5 * (1 - torch.sin(2*torch.pi*x)) * f3
        ret = l1 + l2 + l3 + 0.0*x
        ret = torch.prod(ret, dim=-1, keepdim=self.keepdim)
        return ret


class RandomFeature(TensorMapping):
    def __init__(self, Jn: int, centers: Tensor, radius: float,
                 in_dim: int, out_dim: int=1, bound: float=1.0,
                 activate=torch.tanh, print_status=False):
        """
        @param Jn: int. Number of basis functions at a single center.
        """
        super().__init__()
        Mp, _ = centers.shape
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Jn = Jn

        self.std = StackStd(centers, radius)
        self.linear = MultiLinear(in_dim, out_dim, (Mp, Jn),
                                  dtype=centers.dtype, requires_grad=False)
        self.activate = activate
        self.pou = PoUSin()
        self.um = Parameter(torch.empty((Mp, Jn), dtype=centers.dtype))
        init.normal_(self.um, 0.0, 0.01)

    def number_of_centers(self):
        return self.std.centers.shape[0]

    def number_of_basis(self):
        return self.Jn * self.number_of_centers()

    def number_of_local_basis(self):
        return self.Jn

    def forward(self, p): # (N, 2)
        ret_std: Tensor = self.std(p) # (N, Mp, 2)
        ret = ret_std.unsqueeze(-2) # (N, Mp, 1, 2)
        ret = self.activate(self.linear(ret)) # (N, Mp, Jn, 1)
        ret = torch.einsum('nm, mj, nmjd -> nd', self.pou(ret_std), self.um, ret) # (N, 1)
        return ret


class LocalRandomFeature(TensorMapping):
    def __init__(self, in_dim: int, nf: int, bound: float=1.0, activate=torch.tanh,
                 dtype=None, device=None) -> None:
        """
        @brief Construct a random feature model.

        @param in_dim: int. Dimension of inputs.
        @param nf: int. Number of random features.
        @param bound: float. Bound of uniform distribution to initialize k, b in\
                             the each random feature.
        @param activate: Callable. Activation function after the linear layer.
        @param dtype: torch.dtype. Data type of inputs.
        @param device: torch.device.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.nf = nf
        self.linear = Linear(in_dim, nf, device=device, dtype=dtype)
        self.linear.requires_grad_(False)
        self.set_basis(bound)
        self.activate = activate
        self.uml = Linear(nf, 1, bias=False, device=device, dtype=dtype)
        init.zeros_(self.uml.weight)
        self.device = device

    @property
    def um(self):
        return self.uml.weight

    def set_basis(self, bound):
        init.uniform_(self.linear.weight, -bound, bound)
        init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: Tensor): # (N, d)
        ret = self.activate(self.linear(x)) # (N, nf)
        return self.uml(ret) # (N, 1)

    def number_of_features(self):
        return self.nf

    def basis_val(self, p: Tensor):
        """
        @brief Return values of basis, with shape (N, nf).
        """
        return self.activate(self.linear(p))


class RandomFeatureFlat(TensorMapping):
    def __init__(self, nlrf: int, ngrf: int, centers: Tensor, radius: float,
                 in_dim: int, bound: float=1.0, activate=torch.tanh,
                 print_status=False) -> None:
        """
        @param nlrf: int. Number of local random features.
        @param nglf: int.
        @param centers: 2-d Tensor. Centers of partitions.
        @param radius: float.
        @param in_dim: int. Input dimension.
        @param bound: float. Uniform distribution bound for feature weights and bias.
        @param activate: Callable. The activation function after linear transforms.
        @param print_status: bool.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.nlrf = nlrf
        self.ngrf = ngrf

        self.std = StackStd(centers, radius)
        self.partions: List[LocalRandomFeature] = []
        self.pou = PoUSin(keepdim=True)

        for i in range(self.number_of_partitions()):
            part = LocalRandomFeature(
                    in_dim=in_dim,
                    nf=nlrf,
                    bound=bound,
                    activate=activate,
                    dtype=centers.dtype,
                    device=centers.device
            )
            self.partions.append(part)
            self.add_module(f"part_{i}", part)

        self.gc = torch.mean(centers, dim=0)
        cmax, _ = torch.max(centers, dim=0)
        cmin, _ = torch.min(centers, dim=0)
        self.gr = 0.5 * torch.max(cmax - cmin)
        self.global_ = LocalRandomFeature(
            in_dim=in_dim, nf=ngrf,
            bound=bound, activate=activate,
            dtype=centers.dtype, device=centers.device)

        if print_status:
            print(self.status_string())

    def status_string(self):
        return f"""Random Feature Module
#Partitions: {self.number_of_partitions()},
#Basis(local): {self.number_of_local_basis()},
#Basis(global): {self.number_of_global_basis()},
  - center: {self.gc},
  - radius: {self.gr},
#Basis(total): {self.number_of_basis()},
#Dimension(in): {self.in_dim}"""

    def number_of_partitions(self):
        return self.std.centers.shape[0]

    def number_of_basis(self):
        return self.nlrf * self.number_of_partitions() + self.ngrf

    def number_of_local_basis(self):
        return self.nlrf

    def number_of_global_basis(self):
        return self.ngrf

    @property
    def dtype(self):
        return self.std.centers.dtype

    @property
    def ums(self):
        return [x.um for x in self.partions] + [self.global_.um, ]

    def forward(self, p: Tensor):
        std = self.std(p) # (N, d) -> (N, Mp, d)
        ret = self.global_((p - self.gc[None, :])/self.gr)
        for i in range(self.number_of_partitions()):
            x = std[:, i, :] # (N, d)
            ret += self.partions[i](x) * self.pou(x) # (N, 1)
        return ret # (N, 1)

    def scale(self, p: Tensor, operator: Operator):
        """
        @brief Return the scale by basis and operator.
        """
        MP = self.number_of_partitions()
        std = self.std(p)
        partition_max = torch.zeros((MP, ), dtype=self.dtype)
        for i in range(MP):
            x = std[:, i, :]
            psiphi = self.partions[i].basis_val(p=x) * self.pou(x) # (N, nf)
            partition_max[i] = torch.max(operator(p, psiphi))
        return torch.max(partition_max)

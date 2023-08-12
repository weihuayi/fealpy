"""Modules for the Random Feature Method"""

from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module, init, Linear

from ..nntyping import Operator
from .linear import StackStd
from .module import TensorMapping


class PoU(Module):
    def __init__(self, keepdim=True) -> None:
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: Tensor): # (..., d) -> (..., 1)
        flag = (-1 <= x) * (x < 1)
        flag = torch.prod(flag, dim=-1, keepdim=self.keepdim)
        return flag.to(dtype=x.dtype)


class _PouSin(PoU):
    def forward(self, x: Tensor): # (..., d) -> (..., 1)
        pi = torch.pi
        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        l1 = 0.5 * (1 + torch.sin(2*pi*x)) * f1
        l2 = f2.to(dtype=x.dtype)
        l3 = 0.5 * (1 - torch.sin(2*pi*x)) * f3
        ret = l1 + l2 + l3
        ret = torch.prod(ret, dim=-1, keepdim=self.keepdim)
        return ret


class PoUSin2d(_PouSin):
    """
    @brief Sin-style partition of unity.

    For inputs with shape (..., 2), the output is like (..., ) or (..., 1),\
    and values of each element is between 0 and 1.
    """
    def grad(self, x: Tensor):
        pi = torch.pi

        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        pg = pi * torch.cos(2*pi*x) * f1 - pi * torch.cos(2*pi*x) * f3
        l1 = 0.5 * (1 + torch.sin(2*pi*x)) * f1
        l2 = f2.double()
        l3 = 0.5 * (1 - torch.sin(2*pi*x)) * f3
        p = l1 + l2 + l3
        return pg * p[:, [1, 0]]

    def hessian(self, x: Tensor):
        pi = torch.pi

        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        ph = -2*pi**2 * torch.sin(2*pi*x) * f1 + 2*pi**2 * torch.sin(2*pi*x) * f3
        pg = pi * torch.cos(2*pi*x) * f1 - pi * torch.cos(2*pi*x) * f3
        l1 = 0.5 * (1 + torch.sin(2*pi*x)) * f1
        l2 = f2.double()
        l3 = 0.5 * (1 - torch.sin(2*pi*x)) * f3
        p = l1 + l2 + l3
        hes = torch.zeros((x.shape[0], 2, 2), dtype=x.dtype, device=x.device)
        hes[:, 0, 0] = ph[:, 0] * p[:, 1]
        hes[:, 0, 1] = pg[:, 0] * pg[:, 1]
        hes[:, 1, 0] = pg[:, 0] * pg[:, 1]
        hes[:, 1, 1] = p[:, 0] * ph[:, 1]
        return hes


class RandomFeature2d(TensorMapping):
    def __init__(self, nf: int, bound: float=1.0,
                 dtype=torch.float64, device=None) -> None:
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
        self.out_dim = 1
        self.nf = nf
        self.device = device

        self.linear = Linear(2, nf, device=device, dtype=dtype)
        self.linear.requires_grad_(False)
        self.set_basis(bound)

        self.uml = Linear(nf, 1, bias=False, device=device, dtype=dtype)
        init.zeros_(self.uml.weight)

    @property
    def um(self):
        return self.uml.weight

    def set_um_inplace(self, value: Tensor):
        self.uml.weight.requires_grad_(False)
        self.uml.weight[:] = value
        self.uml.weight.requires_grad_(True)

    def set_basis(self, bound):
        init.uniform_(self.linear.weight, -bound, bound)
        init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: Tensor): # (N, 1)
        ret = torch.cos(self.linear(x)) # (N, nf)
        return self.uml(ret) # (N, 1)

    def number_of_features(self):
        return self.nf

    def basis_val(self, p: Tensor):
        """
        @brief Return values of basis, with shape (N, nf).
        """
        return torch.cos(self.linear(p))

    def basis_grad(self, p: Tensor):
        """
        @brief Return gradient vector of basis, with shape (N, nf, 2).
        """
        a = -torch.sin(self.linear(p))
        return torch.einsum("nf, fx -> nfx", a, self.linear.weight)

    def basis_hessian(self, p: Tensor):
        """
        @brief Return hessian matrix of basis, with shape (N, nf, 2, 2).
        """
        a = -torch.cos(self.linear(p)) * self.linear.weight
        return torch.einsum("nfx, fy -> nfxy", a, self.linear.weight)


class LocalRandomFeature2d(RandomFeature2d):
    """
    @brief Random feature 2d model with units of partitions.
    """
    def __init__(self, nf: int, bound: float = 1, dtype=torch.float64, device=None) -> None:
        super().__init__(nf, bound, dtype, device)
        self.pou = PoUSin2d(keepdim=True)

    def forward(self, p: Tensor): # (N, d)
        ret = torch.cos(self.linear(p)) * self.pou(p)
        return self.uml(ret) # (N, 1)

    def basis_val(self, p: Tensor):
        return torch.cos(self.linear(p)) * self.pou(p)

    def basis_grad(self, p: Tensor):
        l = self.linear(p)
        a = torch.einsum("nd, nf -> nfd", self.pou.grad(p), torch.cos(l))
        b = -self.pou(p)[..., None]\
          * torch.einsum("nf, fd -> nfd", torch.sin(l), self.linear.weight)
        return a + b

    def basis_hessian(self, p: Tensor):
        l = self.linear(p)
        a = torch.einsum("nde, nf -> nfde", self.pou.hessian(p), torch.cos(l))
        b = -2 * torch.einsum("nd, nf, fe -> nfde", self.pou.grad(p),
                              torch.sin(l), self.linear.weight)
        c = -self.pou(p)[..., None, None]\
          * torch.einsum("nf, fd, fe -> nfde", torch.cos(l),
                         self.linear.weight, self.linear.weight)
        return a + b + c


class RandomFeatureFlat(TensorMapping):
    def __init__(self, nlrf: int, centers: Tensor, radius: float,
                 bound: float=1.0, print_status=False) -> None:
        """
        @param nlrf: int. Number of local random features.
        @param centers: 2-d Tensor. Centers of partitions.
        @param radius: float.
        @param bound: float. Uniform distribution bound for feature weights and bias.
        @param print_status: bool.
        """
        super().__init__()
        self.out_dim = 1
        self.nlrf = nlrf

        self.std = StackStd(centers, radius)
        self.partions: List[LocalRandomFeature2d] = []
        self.pou = PoUSin2d(keepdim=True)

        for i in range(self.number_of_partitions()):
            part = LocalRandomFeature2d(
                    nf=nlrf,
                    bound=bound,
                    dtype=centers.dtype,
                    device=centers.device
            )
            self.partions.append(part)
            self.add_module(f"part_{i}", part)

        if print_status:
            print(self.status_string())

    def status_string(self):
        return f"""Random Feature Module
#Partitions: {self.number_of_partitions()},
#Basis(local): {self.number_of_local_basis()},
#Basis(total): {self.number_of_basis()}"""

    def number_of_partitions(self):
        return self.std.centers.shape[0]

    def number_of_basis(self):
        return self.nlrf * self.number_of_partitions()

    def number_of_local_basis(self):
        return self.nlrf

    @property
    def dtype(self):
        return self.std.centers.dtype

    @property
    def ums(self):
        return [x.um for x in self.partions]

    def get_ums(self):
        """
        @brief Get um in each partition as a single tensor with shape (1, M).\
               Where M is number of total basis, equaling to Mp*Jn.
        """
        device = self.get_device()
        ret = torch.zeros((1, self.number_of_basis()), dtype=self.dtype, device=device)
        for idx, part in enumerate(self.partions):
            ret[:, idx*self.nlrf:(idx+1)*self.nlrf] = part.um
        return ret

    def set_ums_inplace(self, value: Tensor):
        """
        @brief Set um in each partition using a single tensor with shape (1, M).\
               Where M is number of total basis, equaling to Mp*Jn.
        """
        for idx, part in enumerate(self.partions):
            part.set_um_inplace(value[:, idx*self.nlrf:(idx+1)*self.nlrf])

    def forward(self, p: Tensor):
        std = self.std(p) # (N, d) -> (N, Mp, d)
        ret = torch.zeros((p.shape[0], 1), dtype=p.dtype, device=p.device)
        for i in range(self.number_of_partitions()):
            x = std[:, i, :] # (N, d)
            ret += self.partions[i](x) # (N, 1)
        return ret # (N, 1)

    def scale(self, p: Tensor, operator: Operator):
        """
        @brief Return the scale by basis and operator.
        @note: This method may need autograd function of PyTorch in `operator`.\
               If the operator object is not based on autograd, this method can\
               not help to get a scale.
        """
        MP = self.number_of_partitions()
        std = self.std(p)
        partition_max = torch.zeros((MP, ), dtype=self.dtype, device=self.get_device())
        for i in range(MP):
            x = std[:, i, :]
            psiphi = self.partions[i].basis_val(p=x) # (N, nf)
            partition_max[i] = torch.max(operator(p, psiphi))
        return torch.max(partition_max)

    def value(self, p: Tensor):
        """
        @brief Return a matrix containing basis values of each sample, with\
               shape (N, M), where M is total local basis.

        @note: This API is designed for the least squares method, therefore the\
               result does not require grad.
        """
        N = p.shape[0]
        M = self.number_of_basis()
        ret = torch.zeros((N, M), dtype=self.dtype, device=self.get_device())
        std = self.std(p)
        for idx, part in enumerate(self.partions):
            x = std[:, idx, :]
            ret[:, idx*self.nlrf:(idx+1)*self.nlrf] = part.basis_val(x)
        return ret

    def grad(self, p: Tensor):
        """
        @brief
        """
        N = p.shape[0]
        M = self.number_of_basis()
        D = p.shape[-1]
        ret = torch.zeros((N, M, D), dtype=self.dtype, device=self.get_device())
        std = self.std(p)
        for idx, part in enumerate(self.partions):
            x = std[:, idx, :]
            ret[:, idx*self.nlrf:(idx+1)*self.nlrf, :] = part.basis_grad(x)/self.std.radius
        return ret

    def hessian(self, p: Tensor):
        """
        @brief
        """
        N = p.shape[0]
        M = self.number_of_basis()
        D = p.shape[-1]
        ret = torch.zeros((N, M, D, D), dtype=self.dtype, device=self.get_device())
        std = self.std(p)
        for idx, part in enumerate(self.partions):
            x = std[:, idx, :]
            ret[:, idx*self.nlrf:(idx+1)*self.nlrf, :, :] = part.basis_hessian(x)/self.std.radius**2
        return ret

    def laplace(self, p: Tensor):
        """
        @brief
        """
        hessian = self.hessian(p)
        return hessian[:, :, 0, 0] + hessian[:, :, 1, 1]

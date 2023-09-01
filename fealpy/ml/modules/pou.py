"""
Partitions of Units
"""
from typing import Union, List, Callable

import torch
from torch.nn import Module
from torch import Tensor, sin, cos

from .linear import Standardize

PI = torch.pi


class _PoU_Fn(Module):
    """
    Base class for PoU functions. These functions apply on each dimension of inputs,
    and in the PoU module, the results will be multiplied in the last dim.

    @note: works on singleton mode.
    """
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def forward(self, x: Tensor):
        raise NotImplementedError

    def flag(self, x: Tensor):
        """
        @brief Support set flag. Return `True` if inside the support set, element-wise.
        """
        raise NotImplementedError

    def dn(self, x: Tensor, order: int) -> Tensor:
        """
        @brief order-specified derivative
        """
        if order == 0:
            return self.forward(x)
        elif order >= 1:
            fn = getattr(self, 'd'+str(order), None)
            if fn is None:
                raise NotImplementedError(f"{order}-order derivative has not been implemented.")
            else:
                return fn(x)
        else:
            raise ValueError(f"order can not be negative, but got {order}.")

    def d1(self, x: Tensor):
        """
        @brief 1-order derivative
        """
        raise NotImplementedError

    def d2(self, x: Tensor):
        """
        @brief 2-order derivative
        """
        raise NotImplementedError

    def d3(self, x: Tensor):
        """
        @brief 3-order derivative
        """
        raise NotImplementedError

    def d4(self, x: Tensor):
        """
        @brief 4-order derivative
        """
        raise NotImplementedError


class PoU(Module):
    """
    Base class for Partitions of Unit (PoU) modules.
    """
    def __init__(self, keepdim=True) -> None:
        super().__init__()
        self.keepdim = keepdim

    def flag(self, x: Tensor) -> Tensor:
        """
        @brief Return a bool tensor with shape (N, ) showing if samples are in\
               the supporting area.
        """
        raise NotImplementedError

    def gradient(self, x: Tensor) -> Tensor:
        """
        @brief Return gradient vector of the function, with shape (N, GD).
        """
        raise NotImplementedError

    def hessian(self, x: Tensor) -> Tensor:
        """
        @brief Return hessian matrix of the function, with shape (N, GD, GD).
        """
        raise NotImplementedError

    def derivative(self, x: Tensor, *idx: int) -> Tensor:
        """
        @brief Return derivatives.
        """
        raise NotImplementedError


class PoUA(PoU):
    def forward(self, x: Tensor): # (..., d) -> (..., 1)
        flag = (-1 <= x) * (x <= 1)
        flag = torch.prod(flag, dim=-1, keepdim=self.keepdim)
        return flag.to(dtype=x.dtype)
        # Here we cast the data type after the operation for lower memory usage.

    def flag(self, x: Tensor):
        flag = ((x >= -1) * (x <= 1))
        return torch.prod(flag, dim=-1, dtype=torch.bool)

    def gradient(self, x: Tensor):
        N, GD = x.shape[0], x.shape[-1]
        return torch.tensor(0, dtype=x.dtype, device=x.device).broadcast_to(N, GD)

    def hessian(self, x: Tensor):
        N, GD = x.shape[0], x.shape[-1]
        return torch.tensor(0, dtype=x.dtype, device=x.device).broadcast_to(N, GD, GD)

    def derivative(self, x: Tensor, *idx: int):
        order = len(idx)
        if order == 0:
            return self.forward(x)
        else:
            N, _ = x.shape[0], x.shape[-1]
            return torch.tensor(0, dtype=x.dtype, device=x.device).broadcast_to(N, 1)


### Sin-style PoU function & module

class _PoU_Sin_Fn(_PoU_Fn):
    def forward(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        l1 = 0.5 * (1 + sin(2*PI*x)) * f1
        l2 = f2.to(dtype=x.dtype)
        l3 = 0.5 * (1 - sin(2*PI*x)) * f3
        return l1 + l2 + l3

    def flag(self, x: Tensor):
        return (x >= -1.25) * (x <= 1.25)

    def d1(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        return PI * cos(2*PI*x) * f1 - PI * cos(2*PI*x) * f3

    def d2(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        return -2*PI**2 * sin(2*PI*x) * f1 + 2*PI**2 * sin(2*PI*x) * f3

    def d3(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        return -4*PI**3 * cos(2*PI*x) * f1 + 4*PI**3 * cos(2*PI*x) * f3

    def d4(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        return 8*PI**4 * sin(2*PI*x) * f1 - 8*PI**4 * sin(2*PI*x) * f3


class PoUSin(PoU):
    """
    @brief Sin-style partition of unity.

    For inputs with shape (..., GD), the output is like (..., ) or (..., 1),\
    and values of each element is between 0 and 1.
    """
    func = _PoU_Sin_Fn()
    def forward(self, x: Tensor): # (..., d) -> (..., 1)
        return torch.prod(self.func(x), dim=-1, keepdim=self.keepdim)

    def flag(self, x: Tensor):
        # As bool type is used in indexing, keepdim is set to False.
        return torch.prod(self.func.flag(x), dim=-1, dtype=torch.bool)

    def gradient(self, x: Tensor):
        pg = self.func.d1(x)
        p = self.func(x)
        N, GD = x.shape[0], x.shape[-1]
        grad = torch.ones((N, GD), dtype=x.dtype)
        for i in range(GD):
            element = torch.zeros((N, GD), dtype=x.dtype)
            element[:] = p[:, i][:, None]
            element[:, i] = pg[:, i]
            grad *= element
        return grad

    def hessian(self, x: Tensor):
        ph = self.func.d2(x)
        pg = self.func.d1(x)
        p = self.func(x)
        N, GD = x.shape[0], x.shape[-1]
        hes = torch.ones((N, GD, GD), dtype=x.dtype)
        for i in range(GD):
            element = torch.zeros((N, GD, GD), dtype=x.dtype)
            element[:] = p[:, i][:, None, None]
            element[:, i, :] = pg[:, i][:, None]
            element[:, :, i] = pg[:, i][:, None]
            element[:, i, i] = ph[:, i]
            hes *= element
        return hes

    def derivative(self, x: Tensor, *idx: int):
        N = x.shape[0]
        os = [0, ] * x.shape[-1]
        for i in idx:
            os[i] += 1

        ret = torch.ones((N, 1), dtype=x.dtype, device=x.device)
        for i in range(x.shape[-1]):
            ret *= self.func.dn(x[:, i:i+1], order=os[i])
        return ret


##################################################
### PoU in Spaces
##################################################

from .function_space import FunctionSpaceBase

class PoULocalSpace(FunctionSpaceBase):
    def __init__(self, pou: PoU, space: FunctionSpaceBase) -> None:
        super().__init__()
        self.pou = pou
        self.space = space

    def flag(self, p: Tensor):
        return self.pou.flag(p)

    def number_of_basis(self) -> int:
        return self.space.number_of_basis()

    def basis(self, p: Tensor) -> Tensor:
        return self.space.basis(p) * self.pou(p)

    def grad_basis(self, p: Tensor) -> Tensor:
        space = self.space
        ret = torch.einsum("nd, nf -> nfd", self.pou.gradient(p), space.basis(p))
        ret += self.pou(p)[..., None] * space.grad_basis(p)
        return ret

    def hessian_basis(self, p: Tensor) -> Tensor:
        space = self.space
        ret = torch.einsum("nxy, nf -> nfxy", self.pou.hessian(p), space.basis(p))
        cross = torch.einsum("nx, nfy -> nfxy", self.pou.gradient(p),
                             space.grad_basis(p))
        ret += cross + torch.transpose(cross, -1, -2)
        ret += self.pou(p)[..., None, None] * space.hessian_basis(p)
        return ret

    def laplace_basis(self, p: Tensor) -> Tensor:
        pass

    def derivative_basis(self, p: Tensor, *idx: int) -> Tensor:
        pass


SpaceFactory = Callable[[], FunctionSpaceBase]

class PoUSpace(FunctionSpaceBase):
    def __init__(self, space_factory: SpaceFactory, centers: Tensor, radius: Tensor,
                 pou: PoU, print_status=False) -> None:
        super().__init__()

        self.std = Standardize(centers=centers, radius=radius)
        self.partitions: List[PoULocalSpace] = []
        self.in_dim = -1
        self.out_dim = -1

        for i in range(self.number_of_partitions()):
            part = PoULocalSpace(pou=pou, space=space_factory())
            if self.in_dim == -1:
                self.in_dim = part.space.in_dim
                self.out_dim = part.space.out_dim
            else:
                if self.in_dim != part.space.in_dim:
                    raise RuntimeError("Can not group together local spaces with"
                                       "different input dimension.")
                if self.out_dim != part.space.out_dim:
                    raise RuntimeError("Can not group together local spaces with"
                                       "differnet output dimension.")

            self.partitions.append(part)
            self.add_module(f"part_{i}", part)

        if print_status:
            print(self.status_string)

    @property
    def status_string(self):
        return f"""PoU Space Group
#Partitions: {self.number_of_partitions()},
#Basis: {self.number_of_basis()}"""

    def number_of_partitions(self):
        return self.std.centers.shape[0]

    def number_of_basis(self):
        return sum(p.number_of_basis() for p in self.partitions)

    @property
    def dtype(self):
        return self.std.centers.dtype

    @property
    def device(self):
        return self.std.centers.device

    def _assemble(self, key: str, p: Tensor) -> Tensor:
        N = p.shape[0]
        M = self.number_of_basis()
        ret = torch.zeros((N, M), dtype=self.dtype, device=self.device)
        std = self.std(p)
        basis_cursor = 0

        for idx, part in enumerate(self.partitions):
            NF = part.number_of_basis()
            x = std[:, idx, :]
            flag = part.flag(x)
            func = getattr(part, key, None)

            if func is None:
                raise ValueError(f"No method named {key} found in the partition.")

            ret[flag, basis_cursor:basis_cursor+NF] += func(x[flag, ...])
            basis_cursor += NF
        return ret

    def basis(self, p: Tensor) -> Tensor:
        """
        @brief
        """
        N = p.shape[0]
        M = self.number_of_basis()
        ret = torch.zeros((N, M), dtype=self.dtype, device=self.device)
        std = self.std(p)
        basis_cursor = 0
        for idx, part in enumerate(self.partitions):
            NF = part.number_of_basis()
            x = std[:, idx, :]
            flag = part.flag(x)
            ret[flag, basis_cursor:basis_cursor+NF] += part.basis(x[flag, ...])
            basis_cursor += NF
        return ret

    def laplace_basis(self, p: Tensor) -> Tensor:
        pass

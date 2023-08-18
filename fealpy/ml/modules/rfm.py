"""Modules for the Random Feature Method"""

from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, init, Linear

from ..nntyping import Operator
from .linear import StackStd
from .module import TensorMapping

PI = torch.pi


################################################################################
### Units of Partitions
################################################################################

class PoU(Module):
    def __init__(self, keepdim=True) -> None:
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: Tensor): # (..., d) -> (..., 1)
        flag = (-1 <= x) * (x <= 1)
        flag = torch.prod(flag, dim=-1, keepdim=self.keepdim)
        return flag.to(dtype=x.dtype)
        # Here we cast the data type after the operation for lower memory usage.

    def flag(self, x: Tensor):
        """
        @brief Return a bool tensor with shape (N, ) showing if samples are in\
               the supporting area.
        """
        flag = ((x >= -1) * (x <= 1))
        return torch.prod(flag, dim=-1, dtype=torch.bool)

    def gradient(self, x: Tensor):
        """
        @brief Return gradient vector of the function, with shape (N, GD).
        """
        N, GD = x.shape[0], x.shape[-1]
        return torch.tensor(0, dtype=x.dtype).broadcast_to(N, GD)

    def hessian(self, x: Tensor):
        """
        @brief Return hessian matrix of the function, with shape (N, GD, GD).
        """
        N, GD = x.shape[0], x.shape[-1]
        return torch.tensor(0, dtype=x.dtype).broadcast_to(N, GD, GD)

### Sin-Style PoU Function

class PoUSin(PoU):
    """
    @brief Sin-style partition of unity.

    For inputs with shape (..., GD), the output is like (..., ) or (..., 1),\
    and values of each element is between 0 and 1.
    """
    def forward(self, x: Tensor): # (..., d) -> (..., 1)
        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        l1 = 0.5 * (1 + torch.sin(2*PI*x)) * f1
        l2 = f2.to(dtype=x.dtype)
        l3 = 0.5 * (1 - torch.sin(2*PI*x)) * f3
        ret = l1 + l2 + l3
        ret = torch.prod(ret, dim=-1, keepdim=self.keepdim)
        return ret

    def flag(self, x: Tensor):
        flag = ((x >= -1.25) * (x <= 1.25))
        return torch.prod(flag, dim=-1, dtype=torch.bool)

    def gradient(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        pg = PI * torch.cos(2*PI*x) * f1 - PI * torch.cos(2*PI*x) * f3
        l1 = 0.5 * (1 + torch.sin(2*PI*x)) * f1
        l2 = f2.to(dtype=x.dtype)
        l3 = 0.5 * (1 - torch.sin(2*PI*x)) * f3
        p = l1 + l2 + l3

        N, GD = x.shape[0], x.shape[-1]
        grad = torch.ones((N, GD), dtype=x.dtype)
        for i in range(GD):
            element = torch.zeros((N, GD), dtype=x.dtype)
            element[:] = p[:, i][:, None]
            element[:, i] = pg[:, i]
            grad *= element
        return grad
        # return pg * p[:, [1, 0]]

    def hessian(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        ph = -2*PI**2 * torch.sin(2*PI*x) * f1 + 2*PI**2 * torch.sin(2*PI*x) * f3
        pg = PI * torch.cos(2*PI*x) * f1 - PI * torch.cos(2*PI*x) * f3
        l1 = 0.5 * (1 + torch.sin(2*PI*x)) * f1
        l2 = f2.to(dtype=x.dtype)
        l3 = 0.5 * (1 - torch.sin(2*PI*x)) * f3
        p = l1 + l2 + l3
        hes = torch.zeros((x.shape[0], 2, 2), dtype=x.dtype, device=x.device)

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
        # hes[:, 0, 0] = ph[:, 0] * p[:, 1]
        # hes[:, 0, 1] = pg[:, 0] * pg[:, 1]
        # hes[:, 1, 0] = pg[:, 0] * pg[:, 1]
        # hes[:, 1, 1] = p[:, 0] * ph[:, 1]

POUTYPE = {
    'sin': PoUSin
}


################################################################################
### Random Feature Models
################################################################################

class RandomFeatureUnit(TensorMapping):
    def __init__(self, in_dim: int, nf: int, bound: Tuple[float, float]=(1.0, PI),
                 dtype=torch.float64, device=None) -> None:
        """
        @brief Construct a random feature model.

        @param in_dim: int. Dimension of inputs.
        @param nf: int. Number of random features.
        @param bound: two floats. Bound of uniform distribution to initialize\
               k, b in the each random feature.
        @param dtype: torch.dtype. Data type of inputs.
        @param device: torch.device.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.nf = nf
        self.device = device

        self.linear = Linear(in_dim, nf, device=device, dtype=dtype)
        self.linear.requires_grad_(False)
        self._set_basis(bound)

        self.uml = Linear(nf, 1, bias=False, device=device, dtype=dtype)
        init.zeros_(self.uml.weight)

    def _set_basis(self, bound: Tuple[float, float]):
        init.uniform_(self.linear.weight, -bound[0], bound[0])
        init.uniform_(self.linear.bias, -bound[1], bound[1])

    @property
    def um(self):
        return self.uml.weight

    def set_um_inplace(self, value: Tensor):
        """
        @brief Set values of um inplace.
        """
        self.uml.weight.requires_grad_(False)
        self.uml.weight[:] = value
        self.uml.weight.requires_grad_(True)

    def forward(self, x: Tensor): # (N, 1)
        ret = torch.cos(self.linear(x)) # (N, nf)
        return self.uml(ret) # (N, 1)

    def number_of_features(self):
        return self.nf

    def basis_value(self, p: Tensor):
        """
        @brief Return values of basis, with shape (N, nf).
        """
        return torch.cos(self.linear(p))

    def basis_gradient(self, p: Tensor):
        """
        @brief Return gradient vector of basis, with shape (N, nf, GD).
        """
        a = -torch.sin(self.linear(p))
        return torch.einsum("nf, fx -> nfx", a, self.linear.weight)

    def basis_hessian(self, p: Tensor):
        """
        @brief Return hessian matrix of basis, with shape (N, nf, GD, GD).
        """
        a = -torch.cos(self.linear(p))
        return torch.einsum("nf, fx, fy -> nfxy", a,
                            self.linear.weight, self.linear.weight)

    def basis_laplace(self, p: Tensor):
        """
        @brief Return basis evaluated by laplace operator, with shape (N, nf).
        """
        a = -torch.cos(self.linear(p))
        return torch.einsum("nf, fd, fd -> nf", a,
                            self.linear.weight, self.linear.weight)

    def basis_derivative(self, p: Tensor, *idx: int):
        """
        @brief Return specified partial derivatives of basis, with shape (N, nf).

        @param *idx: int. index of the independent variable to take partial derivatives.
        """
        order = len(idx)
        if order == 0:
            return torch.cos(self.linear(p))
        elif order == 1:
            a = -torch.sin(self.linear(p))
            return torch.einsum("nf, f -> nf", a, self.linear.weight[: idx[0]])
        elif order == 2:
            a = -torch.cos(self.linear(p))
            return torch.einsum("nf, f, f -> nf", a,
                                self.linear.weight[: idx[0]],
                                self.linear.weight[: idx[1]])
        else:
            raise ValueError("Derivatives higher than order 2 cannot be obtained.")


class LocalRandomFeature(RandomFeatureUnit):
    """
    @brief Random feature in a single Partition.
    """
    def __init__(self, in_dim: int, nf: int, bound: Tuple[float, float]=(1.0, PI),
                 dtype=torch.float64, device=None) -> None:
        super().__init__(in_dim, nf, bound, dtype, device)
        #TODO: Specify PoU function in __init__ parameters.
        self.pou = PoUSin(keepdim=True)

    def forward(self, p: Tensor): # (N, d)
        # We do not use the "input only samples in the support domain" approach,
        # because that would make the computational graph difficult to maintain.
        ret = torch.cos(self.linear(p)) * self.pou(p)
        return self.uml(ret) # (N, 1)

    def flag(self, p: Tensor):
        """
        @brief Return a bool tensor with shape (N,) showing if samples in `p`\
               is in the supporting area.

        @note: For samples outside the supporting area, local random features\
               always outputs zeros.
        """
        return self.pou.flag(p)

    def basis_value(self, p: Tensor):
        return torch.cos(self.linear(p)) * self.pou(p)

    def basis_gradient(self, p: Tensor):
        l = self.linear(p)
        a = torch.einsum("nd, nf -> nfd", self.pou.gradient(p), torch.cos(l))
        b = -self.pou(p)[..., None]\
          * torch.einsum("nf, fd -> nfd", torch.sin(l), self.linear.weight)
        return a + b

    def basis_hessian(self, p: Tensor):
        l = self.linear(p)
        a = torch.einsum("nxy, nf -> nfxy", self.pou.hessian(p), torch.cos(l))
        b = -2 * torch.einsum("nx, nf, fy -> nfxy", self.pou.gradient(p),
                              torch.sin(l), self.linear.weight)
        c = -self.pou(p)[..., None, None]\
          * torch.einsum("nf, fx, fy -> nfxy", torch.cos(l),
                         self.linear.weight, self.linear.weight)
        return a + b + c

    def basis_laplace(self, p: Tensor):
        l = self.linear(p)
        a = torch.einsum("ndd, nf -> nf", self.pou.hessian(p), torch.cos(l))
        b = -2 * torch.einsum("nd, nf, fd -> nf", self.pou.gradient(p),
                              torch.sin(l), self.linear.weight)
        c = -self.pou(p)\
          * torch.einsum("nf, fd, fd -> nf", torch.cos(l),
                         self.linear.weight, self.linear.weight)
        return a + b + c

    def basis_derivative(self, p: Tensor, *idx: int):
        order = len(idx)
        if order == 0:
            return self.basis_value(p)
        elif order == 1:
            l = self.linear(p)
            a = torch.einsum("n, nf -> nf", self.pou.gradient(p)[:, idx[0]], torch.cos(l))
            b = -self.pou(p)\
            * torch.einsum("nf, f -> nf", torch.sin(l), self.linear.weight[:, idx[0]])
            return a + b
        elif order == 2:
            l = self.linear(p)
            a = torch.einsum("n, nf -> nf", self.pou.hessian(p)[:, idx[0], idx[1]], torch.cos(l))
            b = -2 * torch.einsum("n, nf, f -> nf", self.pou.gradient(p)[:, idx[0]],
                                torch.sin(l), self.linear.weight[:, idx[1]])
            c = -self.pou(p)\
            * torch.einsum("nf, f, f -> nf", torch.cos(l),
                            self.linear.weight[:, idx[0]], self.linear.weight[:, idx[1]])
            return a + b + c
        else:
            raise ValueError("Derivatives higher than order 2 cannot be obtained.")


class RandomFeature(TensorMapping):
    def __init__(self, in_dim: int, nlrf: int, centers: Tensor, radius: Union[float, Tensor],
                 bound: Tuple[float, float]=(1.0, PI), print_status=False) -> None:
        """
        @param nlrf: int. Number of local random features.
        @param centers: 2-d Tensor with shape (M, GD). Centers of partitions.
        @param radius: float or Tensor with shape (M,). Radius of partitions.
        @param bound: two floats. Uniform distribution bound for feature weights and bias.
        @param print_status: bool.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.nlrf = nlrf
        if isinstance(radius, float):
            radius = torch.tensor(radius, dtype=centers.dtype).broadcast_to((centers.shape[0],))
        self.std = StackStd(centers, radius)
        self.partions: List[LocalRandomFeature] = []

        for i in range(self.number_of_partitions()):
            part = LocalRandomFeature(
                    in_dim=in_dim,
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
        return self.nlrf * self.std.centers.shape[0]

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
        for idx, part in enumerate(self.partions):
            x = std[:, idx, :]
            flag = part.flag(x) # Only take samples inside the supporting area
            psiphi = self.partions[idx].basis_value(x[flag, ...]) # (N, nf)
            partition_max[idx] = torch.max(operator(p, psiphi))
        return torch.max(partition_max)

    def basis_value(self, p: Tensor):
        """
        @brief Return values of all basis functions.

        @note: This API is designed for the least squares method, therefore the\
               result does not require grad.
        """
        N = p.shape[0]
        M = self.number_of_basis()
        Jn = self.nlrf
        ret = torch.zeros((N, M), dtype=self.dtype, device=self.get_device())
        std = self.std(p)
        for idx, part in enumerate(self.partions):
            x = std[:, idx, :]
            flag = part.flag(x) # Only take samples inside the supporting area
            ret[flag, idx*Jn:(idx+1)*Jn] = part.basis_value(x[flag, ...])
        return ret

    U = basis_value

    def basis_laplace(self, p: Tensor):
        """
        @brief Return values of the Laplacian applied to all basis functions.
        """
        N = p.shape[0]
        M = self.number_of_basis()
        Jn = self.nlrf
        ret = torch.zeros((N, M), dtype=self.dtype, device=self.get_device())
        std = self.std(p)
        for idx, part in enumerate(self.partions):
            x = std[:, idx, :]
            flag = part.flag(x) # Only take samples inside the supporting area
            ret[flag, idx*Jn:(idx+1)*Jn] = part.basis_laplace(x[flag, ...])/self.std.radius[idx]**2
        return ret

    L = basis_laplace

    def basis_derivative(self, p: Tensor, *idx: int):
        """
        @brief Return the partial derivatives of all basis functions\
               with respect to the specified independent variables.
        """
        order = len(idx)
        N = p.shape[0]
        M = self.number_of_basis()
        Jn = self.nlrf
        ret = torch.zeros((N, M), dtype=self.dtype, device=self.get_device())
        std = self.std(p)
        for i, part in enumerate(self.partions):
            x = std[:, i, :]
            flag = part.flag(x) # Only take samples inside the supporting area
            ret[flag, i*Jn:(i+1)*Jn] = part.basis_derivative(x[flag, ...], *idx)/self.std.radius[i]**order
        return ret

    D = basis_derivative

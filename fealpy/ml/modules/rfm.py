"""
Modules for the Random Feature Method
"""

from typing import List, Tuple, Union, Optional

import torch
from torch import Tensor
from torch.nn import init, Linear

from ..nntyping import Operator
from .linear import StackStd
from .module import TensorMapping
from .activate import Activation
from .pou import PoU

PI = torch.pi


################################################################################
### Random Feature Models
################################################################################

class RandomFeatureSpace():
    def __init__(self, in_dim: int, nf: int,
                 activate: Activation,
                 bound: Tuple[float, float]=(1.0, PI),
                 dtype=torch.float64, device=None) -> None:
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
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.nf = nf
        self.dtype = dtype
        self.device = device

        self.linear = Linear(in_dim, nf, device=device, dtype=dtype)
        self.linear.requires_grad_(False)
        self._set_basis(bound)

        self.activate = activate

    def _set_basis(self, bound: Tuple[float, float]):
        init.uniform_(self.linear.weight, -bound[0], bound[0])
        init.uniform_(self.linear.bias, -bound[1], bound[1])

    def number_of_basis(self):
        return self.nf

    def basis_value(self, p: Tensor) -> Tensor:
        """
        @brief Return values of basis, with shape (N, nf).
        """
        return self.activate(self.linear(p))

    def basis_gradient(self, p: Tensor) -> Tensor:
        """
        @brief Return gradient vector of basis, with shape (N, nf, GD).
        """
        a = self.activate.d1(self.linear(p))
        return torch.einsum("nf, fx -> nfx", a, self.linear.weight)

    def basis_hessian(self, p: Tensor) -> Tensor:
        """
        @brief Return hessian matrix of basis, with shape (N, nf, GD, GD).
        """
        a = self.activate.d2(self.linear(p))
        return torch.einsum("nf, fx, fy -> nfxy", a,
                            self.linear.weight, self.linear.weight)

    def basis_laplace(self, p: Tensor) -> Tensor:
        """
        @brief Return basis evaluated by laplace operator, with shape (N, nf).
        """
        a = self.activate.d2(self.linear(p))
        return torch.einsum("nf, fd, fd -> nf", a,
                            self.linear.weight, self.linear.weight)

    def basis_derivative(self, p: Tensor, *idx: int) -> Tensor:
        """
        @brief Return specified partial derivatives of basis, with shape (N, nf).

        @param *idx: int. index of the independent variable to take partial derivatives.
        """
        order = len(idx)
        if order == 0:
            return self.activate(self.linear(p))
        elif order == 1:
            a = self.activate.d1(self.linear(p))
            return torch.einsum("nf, f -> nf", a, self.linear.weight[:, idx[0]])
        elif order == 2:
            a = self.activate.d2(self.linear(p))
            return torch.einsum("nf, f, f -> nf", a,
                                self.linear.weight[:, idx[0]],
                                self.linear.weight[:, idx[1]])
        elif order == 3:
            a = self.activate.d3(self.linear(p))
            return torch.einsum("nf, f, f, f -> nf", a,
                                self.linear.weight[:, idx[0]],
                                self.linear.weight[:, idx[1]],
                                self.linear.weight[:, idx[2]])
        elif order == 4:
            a = self.activate.d4(self.linear(p))
            return torch.einsum("nf, f, f, f, f -> nf", a,
                                self.linear.weight[:, idx[0]],
                                self.linear.weight[:, idx[1]],
                                self.linear.weight[:, idx[2]],
                                self.linear.weight[:, idx[3]])
        raise NotImplementedError("Derivatives higher than order 4 have not been implemented.")


class RFFunction(TensorMapping):
    def __init__(self, space: RandomFeatureSpace, um: Optional[Tensor]) -> None:
        super().__init__()
        dtype = space.dtype
        device = space.device
        M = space.number_of_basis()

        self.space = space
        self.uml = Linear(M, 1, bias=False, device=device, dtype=dtype)
        if um is None:
            init.zeros_(self.uml.weight)
        else:
            self.set_um_inplace(um)

    @property
    def um(self):
        return self.uml.weight

    def set_um_inplace(self, value: Tensor):
        """
        @brief Set values of um inplace.
        """
        with torch.no_grad():
            self.uml.weight[:] = value

    def forward(self, x: Tensor): # (N, 1)
        return self.uml(self.space.basis_value(x)) # (N, 1)


class LocalRandomFeatureSpace(RandomFeatureSpace):
    """
    @brief Random feature space in a single Partition.
    """
    def __init__(self, in_dim: int, nf: int,
                 activate: Activation,
                 pou: PoU,
                 bound: Tuple[float, float]=(1.0, PI),
                 dtype=torch.float64, device=None) -> None:
        super().__init__(in_dim, nf, activate, bound, dtype, device)
        self.pou = pou

    def flag(self, p: Tensor):
        """
        @brief Return a bool tensor with shape (N,) showing if samples in `p`\
               is in the supporting area.

        @note: For samples outside the supporting area, local random features\
               always outputs zeros.
        """
        return self.pou.flag(p)

    def basis_value(self, p: Tensor):
        return super().basis_value(p) * self.pou(p)

    def basis_gradient(self, p: Tensor):
        ret = torch.einsum("nd, nf -> nfd", self.pou.gradient(p), super().basis_value(p))
        ret += self.pou(p)[..., None] * super().basis_gradient(p)
        return ret

    def basis_hessian(self, p: Tensor):
        ret = torch.einsum("nxy, nf -> nfxy", self.pou.hessian(p), super().basis_value(p))
        cross = torch.einsum("nx, nfy -> nfxy", self.pou.gradient(p),
                             super().basis_gradient(p))
        ret += cross + torch.transpose(cross, -1, -2)
        ret += self.pou(p)[..., None, None] * super().basis_hessian(p)
        return ret

    def basis_laplace(self, p: Tensor):
        ret = torch.einsum("ndd, nf -> nf", self.pou.hessian(p), super().basis_value())
        ret += 2 * torch.einsum("nd, nfd -> nf", self.pou.gradient(p),
                                super().basis_gradient(p))
        ret += self.pou(p) * super().basis_laplace(p)
        return ret

    def basis_derivative(self, p: Tensor, *idx: int):
        N = p.shape[0]
        nf = self.number_of_basis()
        order = len(idx)
        ret = torch.zeros((N, nf), dtype=self.dtype, device=self.device)

        if order == 0:
            ret[:] = self.basis_value(p)
        elif order == 1:
            ret += self.pou.derivative(p, idx[0]) * super().basis_value(p)
            ret += self.pou(p) * super().basis_gradient(p, idx[0])
        elif order == 2:
            ret += self.pou.derivative(p, idx[0], idx[1]) * super().basis_value(p)
            ret += self.pou.derivative(p, idx[0]) * super().basis_derivative(p, idx[1])
            ret += self.pou.derivative(p, idx[1]) * super().basis_derivative(p, idx[0])
            ret += self.pou(p) * super().basis_derivative(p, idx[0], idx[1])
        elif order == 3:
            ret += self.pou.derivative(p, idx[0], idx[1], idx[2]) * super().basis_value(p)
            ret += self.pou.derivative(p, idx[0], idx[1]) * super().basis_derivative(p, idx[2])
            ret += self.pou.derivative(p, idx[1], idx[2]) * super().basis_derivative(p, idx[0])
            ret += self.pou.derivative(p, idx[2], idx[0]) * super().basis_derivative(p, idx[1])
            ret += self.pou.derivative(p, idx[0]) * super().basis_derivative(p, idx[2], idx[1])
            ret += self.pou.derivative(p, idx[1]) * super().basis_derivative(p, idx[0], idx[2])
            ret += self.pou.derivative(p, idx[2]) * super().basis_derivative(p, idx[1], idx[0])
            ret += self.pou(p) * super().basis_derivative(p, idx[0], idx[1], idx[2])

        elif order == 4:
            pass
        # TODO: finish this
        else:
            raise NotImplementedError("Derivatives higher than order 4 have bot been implemented.")
        return ret


class RandomFeaturePoUSpace():
    def __init__(self, in_dim: int, nlrf: int, activate: Activation, pou: PoU,
                 centers: Tensor, radius: Union[float, Tensor],
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
        self.partions: List[LocalRandomFeatureSpace] = []

        for i in range(self.number_of_partitions()):
            part = LocalRandomFeatureSpace(
                    in_dim=in_dim,
                    nf=nlrf,
                    activate=activate,
                    pou=pou,
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

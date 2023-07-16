
import torch
from torch import Tensor
from torch.nn import Module, Parameter, init

from .linear import MultiLinear, StackStd
from .module import TensorMapping


class PoU(Module):
    def forward(self, x: Tensor): # (N, Mp, d)
        flag = (-1 <= x) * (x < 1)
        flag = torch.prod(flag, dim=-1)
        return flag.double()


class PoUSin(Module):
    def forward(self, x: Tensor): # (N, Mp, d)
        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        l1 = 0.5 * (1 + torch.sin(2*torch.pi*x)) * f1
        l2 = f2.double()
        l3 = 0.5 * (1 - torch.sin(2*torch.pi*x)) * f3
        ret = l1 + l2 + l3
        ret = torch.prod(ret, dim=-1)
        return ret


class GlobalRandomFeature(TensorMapping):
    def __init__(self, features: int, ni: int, no: int,
                 device=None, dtype=torch.float64) -> None:
        super().__init__()
        self.features = features
        self.l1 = MultiLinear(ni, no, (features, ),
                              device=device, dtype=dtype, requires_grad=False)
        self.um = Parameter(torch.empty(features, device=device, dtype=dtype))
        init.normal_(self.um, 0.0, 0.01)

    def forward(self, p: Tensor):
        ret = torch.cos(self.l1(p))
        return torch.einsum('nbo, b -> no', ret, self.um)


class RandomFeature(TensorMapping):
    def __init__(self, Jn: int, centers: Tensor, radius: float,
                 in_dim: int, out_dim: int=1):
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
        ret = torch.cos(self.linear(ret)) # (N, Mp, Jn, 1)
        ret = torch.einsum('nm, mj, nmjd -> nd', self.pou(ret_std), self.um, ret) # (N, 1)
        return ret


# class RandomFeature(Module):
#     def __init__(self, Jn: int, cs: Tensor, rs: float):
#         super().__init__()
#         self.Jn = Jn
#         self.Mp = cs.shape[0]
#         self.um = torch.empty((self.Mp, Jn), dtype=cs.dtype)
#         self._rfs = []
#         for i in range(self.Mp):
#             rf = LocalRandomFeature(self.um[i, :], Jn, cs[i, :], rs)
#             self._rfs.append(rf)
#             self.add_module(f'{i}', rf)

#     def forward(self, p: Tensor):
#         rets = []
#         for mod in self._rfs: # Mp
#             rets.append(mod(p)) # (N, TD2)
#         ret = torch.stack(rets, dim=1) # (N, Mp, TD2)
#         return torch.sum(ret, dim=1) # (N, TD2)

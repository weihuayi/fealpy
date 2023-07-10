
import torch
from torch import Tensor
from torch.nn import Module, Parameter, init

from .linear import MultiLinear, StackStd
from .module import TensorMapping

class PoU(Module):
    def forward(self, x: Tensor):
        N = x.shape[:-1]
        flag = (-1 < x) * (x < 1)
        flag = torch.cumprod(flag, dim=-1)
        ret = torch.zeros(N, dtype=torch.float64)
        ret[flag] = 1.0
        return ret


class RandomFeature(TensorMapping):
    def __init__(self, Jn: int, centers: Tensor, radius: float):
        """
        @param Jn: int. Number of basis functions at a single center.
        """
        super().__init__()
        Mp, _ = centers.shape
        self.std = StackStd(centers, radius)
        self.linear = MultiLinear(2, 1, (Mp, Jn), dtype=centers.dtype)
        self.pou = PoU()
        self.um = Parameter(torch.empty((Mp, Jn), dtype=centers.dtype))
        init.normal_(self.um, 0.0, 0.1)
        self.Jn = Jn


    def forward(self, p): # (N, 2)
        ret_std: Tensor = self.std(p) # (N, Mp, 2)
        ret = ret_std.unsqueeze(-2) # (N, Mp, 1, 2)
        ret = torch.tanh(self.linear(ret)) # (N, Mp, Jn, 1)
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

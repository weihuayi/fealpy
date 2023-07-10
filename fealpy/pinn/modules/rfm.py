from typing import Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from .linear import MultiLinear, StackStd

class PoU(Module):
    def forward(self, x):
        flag = (x < -1) * (x > 1)
        x[flag] *= 0
        return x


class LocalRandomFeature(Module):
    def __init__(self, um: Tensor, Jn: int, center, radius):
        super().__init__()
        self.linear = MultiLinear(Jn, 2, 1)
        self.std = StackStd(center, radius)
        self.pou = PoU()
        self.um = Parameter(um) # torch.empty((Jn, ), dtype=dtype)

    def forward(self, p):
        ret_std = self.std(p) # (N, Jn, 2)
        ret = torch.tanh(self.linear(ret_std)) # (N, Jn, 1)
        ret = torch.einsum('njd, j -> nd', self.um, ret) # (N, 1)
        return ret * self.PoU(ret_std)


class RandomFeature(Module):
    def __init__(self, Jn: int, cs: Tensor, rs: float):
        super().__init__()
        self.Jn = Jn
        self.Mp = cs.shape[0]
        self.um = torch.empty((self.Mp, Jn), dtype=cs.dtype)
        self._rfs = []
        for i in range(self.Mp):
            rf = LocalRandomFeature(self.um[i, :], Jn, cs[i, :], rs)
            self._rfs.append(rf)
            self.add_module(f'{i}', rf)

    def forward(self, p: Tensor):
        rets = []
        for mod in self._rfs: # Mp
            rets.append(mod(p)) # (N, TD2)
        ret = torch.stack(rets, dim=1) # (N, Mp, TD2)
        return torch.sum(ret, dim=1) # (N, TD2)

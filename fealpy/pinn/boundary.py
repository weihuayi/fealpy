
from typing import Union, Optional, List

import torch
from torch import Tensor

from .nntyping import TensorFunction
from .machine import Solution, ZeroMapping


class _BCSetter():
    _bc: Optional[TensorFunction] = None
    def boundary_condition(self, p:Tensor):
        if self._bc is not None:
            return self._bc(p)
        else:
            raise NotImplementedError

    def set_bc(self, fn: TensorFunction):
        """
        @brief Set boundary condition.
        """
        self._bc = fn
        return fn


class _LSSetter():
    _ls: Optional[TensorFunction] = None
    def level_set(self, p:Tensor):
        if self._ls is not None:
            return self._ls(p)
        else:
            raise NotImplementedError

    def set_level_set(self, fn: TensorFunction):
        """
        @brief Use a level-set function to locate the boundary.
        """
        self._ls = fn
        return fn


class _BoxSetter():
    _box: List[float] = []
    def boundary_box(self):
        if len(self._box) >= 2:
            assert len(self._box) % 2 == 0
            return self._box
        else:
            return ValueError

    def set_box(self, box: List[float]):
        """
        @brief Using a box boundary.
        """
        self._box = list(box)


class LevelSetDBCSolution(Solution, _BCSetter, _LSSetter):
    """
    @brief A solution of problems with dirichlet boundary conditions, and the boundary\
           is given through a level-set function. Use the decorators `@set_bc` and\
           `@set_level_set` to set the boundary infomation before training.
    """
    def forward(self, p: Tensor):
        lsv = self.level_set(p)
        return -lsv * self.__net(p) + (1+lsv) * self.boundary_condition(p)


def mkf(length: int, *inputs: Union[Tensor, float]):
    """Make features"""
    ret = torch.zeros((length, len(inputs)), dtype=torch.float32)
    for i, item in enumerate(inputs):
        if isinstance(item, Tensor):
            ret[:, i] = item[:, 0]
        else:
            ret[:, i] = item
    return ret


class _2dSpaceTime(Solution):
    _lef: TensorFunction = ZeroMapping()
    _le: float = 0
    _ref: Optional[TensorFunction] = None
    _re: float = 1
    _if: Optional[TensorFunction] = None

    def set_left_edge(self, x: float=0.0):
        self._le = float(x)
        return self._set_lef
    def _set_lef(self, bc: TensorFunction):
        self._lef = bc
        return bc
    def set_right_edge(self, x: float=1.0):
        self._re = float(x)
        return self._set_ref
    def _set_ref(self, bc: TensorFunction):
        self._ref = bc
        return bc
    def set_initial(self, ic: TensorFunction):
        self._if = ic
        return ic


class TFC2dSpaceTimeDirichletBC(_2dSpaceTime):
    """
    Solution on space(1d)-time(1d) domain with dirichlet boundary conditions,
    based on Theory of Functional Connections.
    """
    def forward(self, p: Tensor) -> Tensor:
        m = p.shape[0]
        l = self._re - self._le
        t = p[..., 0:1]
        x = p[..., 1:2]
        x_0 = torch.cat([torch.zeros_like(t), x], dim=1)
        t_0 = torch.cat([t, torch.ones_like(x)*self._le], dim=1)
        t_1 = torch.cat([t, torch.ones_like(x)*self._re], dim=1)
        return self._if(x)\
            + (self._re-x)/l * (self._lef(t)-self._if(mkf(m, self._le)))\
            + (x-self._le)/l * (self._ref(t)-self._if(mkf(m, self._re)))\
            + self.net(p) - self.net(x_0)\
            - (self._re-x)/l * (self.net(t_0)-self.net(mkf(m, 0, self._le)))\
            - (x-self._le)/l * (self.net(t_1)-self.net(mkf(m, 0, self._re)))


from typing import Union, Optional, List

import torch
from torch import Tensor
from torch.nn import Module

from ..nntyping import TensorFunction
from ..tools import mkfs
from .module import Solution, ZeroMapping, Projected


### Boundary setting tools

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
    GD: int = 0
    def boundary_box(self):
        if len(self._box) >= 2:
            assert len(self._box) % 2 == 0
            return self._box
        raise ValueError('Use `set_box` method to set boundary before using.')

    def set_box(self, box: List[float]):
        """
        @brief Using a box boundary.

        @param box: a list containing floats to specify bounds in each axis.\
                    For example, `[0, 1, 0, 1]`.
        """
        assert len(box) >= 2
        assert len(box) % 2 == 0
        self._box = list(box)
        self.GD = len(box) // 2


### TFC applications

class LevelSetDBCSolution(Solution, _BCSetter, _LSSetter):
    """
    @brief A solution of problems with dirichlet boundary conditions, and the boundary\
           is given through a level-set function. Use the decorators `@set_bc` and\
           `@set_level_set` to set the boundary infomation before training.
    """
    def forward(self, p: Tensor):
        lsv = self.level_set(p)
        return -lsv * self.__net(p) + (1+lsv) * self.boundary_condition(p)


class BoxDBCSolution(Solution, _BCSetter, _BoxSetter):
    """
    @brief A solution of problems with dirichlet boundary conditions in a box area.\
           This is based on Theory of Functional Connection in 1d, 2d and 3d.

    @note !Not Fully Implemented! Now only 1d and 2d are supported.
    """
    def __init__(self, net: Optional[Module] = None, time_idx: Optional[int]=None) -> None:
        super().__init__(net)
        self._time_idx = time_idx

    def _space_fn(self, p: Tensor):
        if self._time_idx is None:
            return self.__net
        else:
            comps: List[Optional[Tensor]] = [None] * p.shape[-1]
            comps[self._time_idx] = p[..., self._time_idx:self._time_idx+1]
            return Projected(self.__net, comps=comps)

    def forward(self, p: Tensor):
        shape = p.shape[:-1] + (1,)
        up = self.net(p)
        shape_m = (3,)*self.GD + up.shape

        if self.GD != p.shape[-1]:
            raise ValueError('Geometry dimension of inputs mismatch the box boundary.')

        u = self.net
        c = self.boundary_condition
        b = self.boundary_box()
        M = torch.zeros(shape_m, dtype=p.dtype, device=p.device)

        if self.GD == 1:
            c1 = mkfs(b[0], f_shape=shape, device=p.device)
            c2 = mkfs(b[1], f_shape=shape, device=p.device)
            M[0, ...] = up
            M[1, ...] = c(c1) - u(c1)
            M[2, ...] = c(c2) - u(c2)

            lp = b[1] - b[0]
            vp = mkfs(1, (b[1]-p)/lp, (p-b[0])/lp)

            return torch.einsum("i...f, ...i -> ...f", M, vp)

        elif self.GD == 2:
            x, y = torch.split(p, 1, dim=-1)
            y_1 = mkfs(b[0], y)
            y_2 = mkfs(b[1], y)
            x_1 = mkfs(x, b[2])
            x_2 = mkfs(x, b[3])
            c_1 = mkfs(b[0], b[2], f_shape=shape, device=p.device)
            c_2 = mkfs(b[1], b[2], f_shape=shape, device=p.device)
            c_3 = mkfs(b[0], b[3], f_shape=shape, device=p.device)
            c_4 = mkfs(b[1], b[3], f_shape=shape, device=p.device)

            M[0, 0, ...] = up
            M[0, 1, ...] = c(x_1) - u(x_1)
            M[0, 2, ...] = c(x_2) - u(x_2)
            M[1, 0, ...] = c(y_1) - u(y_1)
            M[1, 1, ...] = u(c_1) - c(c_1)
            M[1, 2, ...] = u(c_3) - c(c_3)
            M[2, 0, ...] = c(y_2) - u(y_2)
            M[2, 1, ...] = u(c_2) - c(c_2)
            M[2, 2, ...] = u(c_4) - c(c_4)

            lx = b[1] - b[0]
            ly = b[3] - b[2]

            vx = mkfs(1, (b[1]-x)/lx, (x-b[0])/lx)
            vy = mkfs(1, (b[3]-y)/ly, (y-b[2])/ly)

            # if getattr(self, 'einsum_path', None) is None:
            #     self.einsum_path = torch.einsum

            return torch.einsum("ij...f, ...i, ...j -> ...f", M, vx, vy)

        elif self.GD == 3:
            raise NotImplementedError


### old implementations
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

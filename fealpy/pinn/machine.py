from typing import List, Tuple, Optional, Union

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.autograd import Variable

from .sampler import Sampler
from .nntyping import TensorFunction, VectorFunction, Operator


class Solution(Module):
    def __init__(self, net: Module) -> None:
        super().__init__()
        self.__net = net

    @property
    def net(self):
        return self.__net

    def forward(self, p: Tensor):
        return self.__net(p)

    def from_cell_bc(self, bc: NDArray, mesh) -> NDArray:
        """
        From bc in mesh cells to outputs of the solution.

        Return
        ---
        outputs: 2-D Array. Outputs in every integral points and every cells.
        """
        points = mesh.cell_bc_to_point(bc)
        points_tensor = torch.tensor(points, dtype=torch.float32)
        return self.forward(points_tensor).detach().numpy()

    def estimate_error(self, other: VectorFunction, space):
        """
        Calculate error between the solution and `other` in finite element space `space`.

        Parameters
        ---
        other: VectorFunction.
        space: FiniteElementSpace.

        Return
        ---
        error: float.
        """
        def f(bc):
            val = self.from_cell_bc(bc, space.mesh)
            return (val - other(bc))**2

        return np.sqrt(space.integralalg.integral(f, False))

    def meshgrid_mapping(self, *xi: NDArray):
        """
        Parameters
        ---
        *xi: ArrayLike.
            See `numpy.meshgrid`.

        Return
        ---
        outputs, (X1, X2, ..., Xn)
        """
        mesh = np.meshgrid(*xi)
        flat_mesh = [np.ravel(x).reshape(-1, 1) for x in mesh]
        mesh_pt = [Variable(torch.from_numpy(x).float(), requires_grad=True) for x in flat_mesh]
        pt_u: torch.Tensor = self.forward(torch.cat(mesh_pt, dim=1))
        u_plot: NDArray = pt_u.data.cpu().numpy()
        return u_plot.reshape(mesh[0].shape), mesh


class LearningMachine():
    """Neural network trainer."""
    def __init__(self, s: Solution, optimizer: Optimizer,
                 cost_function: Optional[Module]=None) -> None:
        self.__solution = s
        self.optimizer = optimizer

        if cost_function:
            self.cost = cost_function
        else:
            self.cost = torch.nn.MSELoss(reduction='mean')

    @property
    def solution(self):
        return self.__solution


    def loss(self, sampler: Sampler, func: Operator,
             target: Optional[Tensor]=None) -> Tensor:
        """
        Calculate loss value.

        Args
        ---
        sampler: Sampler.
        func: Operator.
            A function get x and u(x) as args. (e.g A pde or boundary condition.)
        target: Tensor or None.
            If `None`, the output will be compared with zero tensor.

        Notes
        ---
        Arg `func` should be defined like:
        ```
            def equation(p: Tensor, u: TensorFunction) -> Tensor:
                ...
        ```
        Here `u` may be a function of `p`.
        """
        inputs = sampler.run()
        outputs = func(inputs, self.solution.forward)
        if target:
            return self.cost(outputs, target)
        else:
            return self.cost(outputs, torch.zeros_like(outputs))


def mkf(length: int, *inputs: torch.Tensor):
    """Make features"""
    ret = torch.zeros((length, len(inputs)), dtype=torch.float32)
    for i, item in enumerate(inputs):
        if isinstance(item, torch.Tensor):
            ret[:, i] = item[:, 0]
        else:
            ret[:, i] = item
    return ret


class _2dSpaceTime(Solution):
    _lef: Optional[TensorFunction] = None
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
    def forward(self, p: torch.Tensor) -> torch.Tensor:
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

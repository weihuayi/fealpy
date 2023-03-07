from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.autograd import Variable

from .sampler import Sampler
from .nntyping import TensorFunction, VectorFunction


class LearningMachine():
    """Neural network trainer."""
    def __init__(self, net: Module, optimizer: Optimizer,
                 cost_function: Optional[Module]=None) -> None:
        self.__net = net
        self.optimizer = optimizer
        self.loss_list: List[Tuple[float, TensorFunction, Sampler]] = []
        self.loss_summary = torch.zeros((1, ))

        if cost_function:
            self.cost = cost_function
        else:
            self.cost = torch.nn.MSELoss(reduction='mean')

    @property
    def net(self):
        return self.__net

    @property
    def loss_data(self):
        return self.loss_summary.data.numpy()

    def function(self, p: torch.Tensor) -> torch.Tensor:
        """
        Define how the neural network `self.net` constituts the solution. This defaults to

        ```
            def function(self, p):
                return self.net(p)
        ```

        Override this method to customize.
        """
        return self.net(p)

    def add_loss(self, coef: float, func: TensorFunction, sampler: Sampler):
        self.loss_list.append([coef, func, sampler])

    def backward(self):
        self.optimizer.zero_grad()
        self.loss_summary = torch.zeros((1, ))

        for coef, func, sampler in self.loss_list:
            ret = func(sampler.run())
            loss = self.cost(ret, torch.zeros((sampler.m, 1)))
            self.loss_summary = coef*loss + self.loss_summary
        self.loss_summary.backward()

    def step(self):
        self.optimizer.step()

    def iterations(self, n_iter: int):
        """A for-loop for training the neural network. This include `backward()` and `step()`.

        Example
        ---
        ```
            for epoch in lm.iterations(1000):
                if epoch % 100 == 0:
                    print(f"Epoch: {epoch}, Loss: {lm.loss_data}")
        ```
        """
        for i in range(n_iter):
            self.backward()
            self.step()
            yield i

    def from_cell_bc(self, bc: NDArray, mesh) -> NDArray:
        """
        From bc in mesh cells to outputs of the network.

        Return
        ---
        outputs: 2-D Array. Outputs in every integral points and every cells.
        """
        points = mesh.cell_bc_to_point(bc)
        points_tensor = torch.tensor(points, dtype=torch.float32)
        return self.function(points_tensor).detach().numpy()

    def estimate_error(self, other: VectorFunction, space):
        """
        Calculate error between the network and `other` in finite element space `space`.

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
        pt_u: torch.Tensor = self.function(torch.cat(mesh_pt, dim=1))
        u_plot: NDArray = pt_u.data.cpu().numpy()
        return u_plot.reshape(mesh[0].shape), mesh


def mkf(length: int, *inputs: torch.Tensor):
    """Make features"""
    ret = torch.zeros((length, len(inputs)), dtype=torch.float32)
    for i, item in enumerate(inputs):
        if isinstance(item, torch.Tensor):
            ret[:, i] = item[:, 0]
        else:
            ret[:, i] = item
    return ret


class _2dSpaceTime(LearningMachine):
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
    Learning Machine working on
    space(1d)-time(1d) domain with dirichlet boundary conditions, based on Theory of Functional Connections.
    """
    def function(self, p: torch.Tensor) -> torch.Tensor:
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

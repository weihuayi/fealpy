from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch.nn import Module
from torch.optim import Optimizer

from .sampler import Sampler
from .nntyping import TensorFunction, VectorFunction


class LearningMachine():
    cost = torch.nn.MSELoss(reduction='mean')
    def __init__(self, net: Module, optimizer: Optimizer) -> None:
        self.net = net
        self.optimizer = optimizer
        self.loss_list: List[Tuple[float, TensorFunction, Sampler]] = []
        self.loss_summary = torch.zeros((1, ))

    @property
    def loss_data(self):
        return self.loss_summary.data.numpy()

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

    def from_cell_bc(self, bc: NDArray, mesh) -> NDArray:
        """
        From bc in mesh cells to outputs of the network.

        Return
        ---
        outputs: 2-D Array. Outputs in every integral points and every cells.
        """
        points = mesh.cell_bc_to_point(bc)
        points_tensor = torch.tensor(points, dtype=torch.float32)
        return self.net(points_tensor).detach().numpy()

    def estimate_error(self, other: VectorFunction, space):
        """
        Calculate differences between the network and `other` in finite element space `space`.

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

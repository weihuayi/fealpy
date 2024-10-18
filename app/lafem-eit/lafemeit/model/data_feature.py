
from typing import Callable, Optional

from torch import Tensor
import torch.nn as nn

from ..solver import LaplaceFEMSolver


TensorFunc = Callable[[Tensor], Tensor]


class DataPreprocessor(nn.Module):
    """Data Feature Boundary Solver based on FEM."""
    def __init__(self, solver: LaplaceFEMSolver) -> None:
        super().__init__()
        self.solver = solver
        self.vuh = None

    __call__: TensorFunc

    def forward(self, input: Tensor) -> Tensor:
        BATCH, CHANNEL, _, NNBD = input.shape
        solver = self.solver

        # NOTE: Merge the Batch and Channel axis to vectorize in the FDM solver.
        input = input.reshape(-1, 2, NNBD) # [B*CH, 2, NN_bd]
        gd, gn = input[:, 0, :], input[:, 1, :] # [B*CH, NN_bd]
        vuh = solver.solve_from_potential(gd) # [B*CH, gdof]
        vn = solver.normal_derivative(vuh) # [B*CH, bddof]
        gnvn = gn - vn # [B*CH, bddof]
        self.vuh = vuh

        return gnvn.reshape(BATCH, CHANNEL, -1) # [B, CH, bddof]


class DataFeature(nn.Module):
    """Data Feature Solver based on FEM."""
    def __init__(self, solver: LaplaceFEMSolver,
                 bc_filter: Optional[TensorFunc]=None) -> None:
        super().__init__()
        self.solver = solver
        self.bc_filter = bc_filter

    __call__: TensorFunc

    def forward(self, input: Tensor) -> Tensor:
        BATCH, CHANNEL, NNBD = input.shape
        solver = self.solver

        if self.bc_filter is not None:
            gnvn = self.bc_filter(input) # [B, CH, NN_bd]
        else:
            gnvn = input

        # NOTE: Merge the Batch and Channel axis again for the FDM solver.
        gnvn = gnvn.reshape(-1, NNBD) # [B*CH, NN_bd]
        val = self.solver.solve_from_current(gnvn) # [B*CH, gdof]
        img = solver.value_on_nodes(val) # [B*CH, NN]
        self.img = img

        # NOTE: Return the result shape
        return img.reshape(BATCH, CHANNEL, -1) # [B, CH, NN]

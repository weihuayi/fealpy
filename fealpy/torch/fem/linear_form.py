
from typing import TypeVar, Optional, List

import torch
from torch import Tensor

from .. import logger
from ..functionspace.space import FunctionSpace
from .integrator import Integrator
from .form import Form


_FS = TypeVar('_FS', bound=FunctionSpace)


class LinearForm(Form[_FS]):
    r"""@brief"""
    def __init__(self, space: _FS, retain_ints: bool=False, batch_size: int=0):
        self.space = space
        self.dintegrators: List[Integrator] = []
        self.bintegrators: List[Integrator] = []
        self._M: Optional[Tensor] = None
        self.retain_ints = retain_ints
        self.batch_size = batch_size

    def _single_assembly(self) -> Tensor:
        space = self.space
        device = space.device
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()
        cell2dof = space.cell_to_dof()
        NC = cell2dof.shape[0]
        global_vec_shape = (gdof,)

        if len(self.dintegrators) > 0:
            local_vec_shape = (NC, ldof)
            cell_vec = torch.zeros(local_vec_shape, dtype=space.ftype, device=device)

            for i in range(len(self.dintegrators)):
                di = self.dintegrators[i]
                cell_vec = cell_vec + di.assembly(space)

                if not self.retain_ints:
                    di.clear()

            indices = cell2dof.ravel().unsqueeze(0)
            V = torch.sparse_coo_tensor(indices, cell_vec.ravel(), global_vec_shape)

        else:
            V = torch.sparse_coo_tensor(
                torch.empty((1, 0), dtype=space.itype, device=device),
                torch.empty((0,), dtype=space.ftype, device=device),
            )

        for i in range(len(self.bintegrators)):
            bi = self.bintegrators[i]
            V = V + bi.assembly(space)

            if not self.retain_ints:
                bi.clear()

        self._V = V.coalesce()
        logger.info(f"Linear form vector constructed, with shape {list(V.shape)}.")

        return self._V

    def _batch_assembly(self) -> Tensor:
        space = self.space
        device = space.device
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()
        cell2dof = space.cell_to_dof()
        NC = cell2dof.shape[0]
        batch = self.batch_size
        global_vec_shape = (gdof, batch)

        if len(self.dintegrators) > 0:
            local_vec_shape = (NC, ldof, batch)
            cell_vec = torch.zeros(local_vec_shape, dtype=space.ftype, device=device)

            for i in range(len(self.dintegrators)):
                di = self.dintegrators[i]
                new_vec = di.assembly(space)

                if new_vec.ndim == 2:
                    new_vec = new_vec.unsqueeze(-1).expand(local_vec_shape)

                cell_vec = cell_vec + new_vec

                if not self.retain_ints:
                    di.clear()

            indices = cell2dof.ravel().unsqueeze(0)
            cv_flatten = cell_vec.reshape(-1, batch)
            V = torch.sparse_coo_tensor(indices, cv_flatten, global_vec_shape)

        else:
            V = torch.sparse_coo_tensor(
                torch.empty((1, 0), dtype=space.itype, device=device),
                torch.empty((0, batch), dtype=space.ftype, device=device),
            )

        for i in range(len(self.bintegrators)):
            bi = self.bintegrators[i]
            new_vec = bi.assembly(space)
            values = new_vec.values()

            if values.ndim == 1:
                values = values.unsqueeze(-1).expand(-1, batch)
                new_vec = torch.sparse_coo_tensor(
                    values.indices(), values, global_vec_shape
                )

            V = V + new_vec

            if not self.retain_ints:
                bi.clear()

        self._V = V.coalesce()
        logger.info(f"Linear form vector constructed, with shape {list(V.shape)}.")

        return self._V

    def assembly(self) -> Tensor:
        r"""Assembly the linear form vector. Returns COO Tensor of shape (gdof,)."""
        if self.batch_size == 0:
            return self._single_assembly()
        elif self.batch_size > 0:
            return self._batch_assembly()
        else:
            raise ValueError("batch_size must be a non-negative integer.")

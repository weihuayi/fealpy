
from typing import TypeVar, Optional, List

import torch
from torch import Tensor

from .. import logger
from ..functionspace.space import FunctionSpace
from .integrator import Integrator
from .form import Form


_FS = TypeVar('_FS', bound=FunctionSpace)


class BilinearForm(Form[_FS]):
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
        global_mat_shape = (gdof, gdof)

        if len(self.dintegrators) > 0:
            local_mat_shape = (NC, ldof, ldof)
            cell_mat = torch.zeros(local_mat_shape, dtype=space.ftype, device=device)

            for i in range(len(self.dintegrators)):
                di = self.dintegrators[i]
                cell_mat = cell_mat + di.assembly(space)
                if not self.retain_ints:
                    di.clear()

            I = torch.broadcast_to(cell2dof[:, :, None], size=local_mat_shape)
            J = torch.broadcast_to(cell2dof[:, None, :], size=local_mat_shape)
            indices = torch.stack([I.ravel(), J.ravel()], dim=0)
            M = torch.sparse_coo_tensor(indices, cell_mat.ravel(), size=global_mat_shape)

        else: # Create an empty global matrix if no domain integrators.
            M = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=space.itype, device=device),
                torch.empty((0,), dtype=space.ftype, device=device),
                size=global_mat_shape
            )

        for i in range(len(self.bintegrators)):
            bi = self.bintegrators[i]
            M = M + bi.assembly(space)
            if not self.retain_ints:
                bi.clear()

        return M

    def _batch_assembly(self) -> Tensor:
        space = self.space
        device = space.device
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()
        cell2dof = space.cell_to_dof()
        NC = cell2dof.shape[0]
        batch = self.batch_size
        assert batch > 0
        global_mat_shape = (gdof, gdof, batch)

        if len(self.dintegrators) > 0:
            local_mat_shape = (NC, ldof, ldof, batch)
            cell_mat = torch.zeros(local_mat_shape, dtype=space.ftype, device=device)

            for i in range(len(self.dintegrators)):
                di = self.dintegrators[i]
                new_mat = di.assembly(space)

                if new_mat.ndim == 3: # If the matrix does not have batch dimension
                    new_mat = new_mat.unsqueeze(-1).expand(local_mat_shape)
                # Then, the batch dimension should be exactly the same
                cell_mat = cell_mat + new_mat

                if not self.retain_ints:
                    di.clear()

            I = torch.broadcast_to(cell2dof[:, :, None], size=local_mat_shape)
            J = torch.broadcast_to(cell2dof[:, None, :], size=local_mat_shape)
            indices = torch.stack([I.ravel(), J.ravel()], dim=0)
            cm_flatten = cell_mat.reshape(-1, batch)
            M = torch.sparse_coo_tensor(indices, cm_flatten, size=global_mat_shape)

        else: # Create an empty global matrix if no domain integrators.
            M = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=space.itype, device=device),
                torch.empty((0, batch), dtype=space.ftype, device=device),
                size=global_mat_shape
            )

        for i in range(len(self.bintegrators)):
            bi = self.bintegrators[i]
            new_mat = bi.assembly(space)
            value = new_mat.values()

            if value.ndim == 1:
                value = value[..., None].expand(-1, batch)
                new_mat = torch.sparse_coo_tensor(
                    new_mat.indices(), value, size=global_mat_shape
                )

            M = M + new_mat

            if not self.retain_ints:
                bi.clear()

        return M

    def assembly(self, coalesce=True) -> Tensor:
        r"""Assembly the bilinear form matrix. Returns COO Tensor of shape (gdof, gdof)."""
        if self.batch_size == 0:
            M = self._single_assembly()
        elif self.batch_size > 0:
            M = self._batch_assembly()
        else:
            raise ValueError("batch_size must be a non-negative integer.")

        self._M = M.coalesce() if coalesce else M
        logger.info(f"Bilinear form matrix constructed, with shape {list(self._M.shape)}.")

        return self._M

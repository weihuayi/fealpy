
from typing import TypeVar, Optional

import torch
from torch import Tensor

from .. import logger
from ..functionspace.space import FunctionSpace
from .form import Form


_FS = TypeVar('_FS', bound=FunctionSpace)


class BilinearForm(Form[_FS]):
    r"""@brief"""
    def __init__(self, space: _FS, batch_size: int=0):
        self.space = space
        self.integrators = {}
        self.memory = {}
        self._M: Optional[Tensor] = None
        self.batch_size = batch_size

    def check_local_shape(self, entity_to_global: Tensor, local_tensor: Tensor):
        if entity_to_global.ndim != 2:
            raise ValueError("entity-to-global relationship should be a 2D tensor, "
                             f"but got shape {tuple(entity_to_global.shape)}.")
        if entity_to_global.shape[0] != local_tensor.shape[0]:
            raise ValueError(f"entity_to_global.shape[0] != local_tensor.shape[0]")
        if local_tensor.ndim not in (3, 4):
            raise ValueError("Output of operator integrators should be 3D "
                             "(or 4D with batch in the last dimension), "
                             f"but got shape {tuple(local_tensor.shape)}.")
        ldof = self.space.number_of_local_dofs()
        if local_tensor.shape[1:3] != (ldof, ldof):
            raise ValueError(f"Size of operator integrator outputs on the 1 and 2 "
                             f"dimension should equal to the number of local dofs ({ldof}), "
                             f"but got shape {tuple(local_tensor.shape)}.")

    def _single_assembly(self, retain_ints: bool) -> Tensor:
        space = self.space
        device = space.device
        gdof = space.number_of_global_dofs()
        global_mat_shape = (gdof, gdof)
        M = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=space.itype, device=device),
            torch.empty((0,), dtype=space.ftype, device=device),
            size=global_mat_shape
        )

        for group, INTS in self.integrators.items():
            e2dof = INTS[0].to_global_dof(space)
            group_tensor = self.assembly_group(group, retain_ints)
            I = torch.broadcast_to(e2dof[:, :, None], size=group_tensor.shape)
            J = torch.broadcast_to(e2dof[:, None, :], size=group_tensor.shape)
            indices = torch.stack([I.ravel(), J.ravel()], dim=0)
            M += torch.sparse_coo_tensor(indices, group_tensor.ravel(), size=global_mat_shape)

        return M

    def _batch_assembly(self, retain_ints: bool, batch_size: int) -> Tensor:
        space = self.space
        device = space.device
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()
        global_mat_shape = (gdof, gdof, batch_size)
        M = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=space.itype, device=device),
            torch.empty((0, batch_size), dtype=space.ftype, device=device),
            size=global_mat_shape
        )

        for group, INTS in self.integrators.items():
            e2dof = INTS[0].to_global_dof(space)
            NC = e2dof.size(0)
            local_mat_shape = (NC, ldof, ldof, batch_size)
            group_tensor = self.assembly_group(group, retain_ints)

            if group_tensor.ndim == 3:
                group_tensor = group_tensor.unsqueeze_(-1).expand(local_mat_shape)

            I = torch.broadcast_to(e2dof[:, :, None], size=group_tensor.shape)
            J = torch.broadcast_to(e2dof[:, None, :], size=group_tensor.shape)
            indices = torch.stack([I.ravel(), J.ravel()], dim=0)
            group_tensor = group_tensor.reshape(-1, batch_size)
            M += torch.sparse_coo_tensor(indices, group_tensor, size=global_mat_shape)

        return M

    def assembly(self, coalesce=True, retain_ints: bool=False) -> Tensor:
        """Assembly the bilinear form matrix. Returns COO Tensor of shape (gdof, gdof)."""
        if self.batch_size == 0:
            M = self._single_assembly(retain_ints)
        elif self.batch_size > 0:
            M = self._batch_assembly(retain_ints, self.batch_size)
        else:
            raise ValueError("batch_size must be a non-negative integer.")

        self._M = M.coalesce() if coalesce else M
        logger.info(f"Bilinear form matrix constructed, with shape {list(self._M.shape)}.")

        return self._M

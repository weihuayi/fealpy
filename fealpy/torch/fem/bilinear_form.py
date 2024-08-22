
from typing import Optional

import torch
from torch import Tensor

from .. import logger
from .form import Form


class BilinearForm(Form):
    def check_local_shape(self, entity_to_global: Tensor, local_tensor: Tensor):
        if entity_to_global.ndim != 2:
            raise ValueError("entity-to-global relationship should be a 2D tensor, "
                             f"but got shape {tuple(entity_to_global.shape)}.")
        if entity_to_global.shape[0] != local_tensor.shape[0]:
            raise ValueError(f"entity_to_global.shape[0] != local_tensor.shape[0]")
        if local_tensor.ndim not in (3, 4):
            raise ValueError("Output of operator integrators should be 3D "
                             "(or 4D with batch in the first dimension), "
                             f"but got shape {tuple(local_tensor.shape)}.")

    def check_space(self):
        if len(self._spaces) not in {1, 2}:
            raise ValueError("BilinearForm should have 1 or 2 spaces, "
                             f"but got {len(self._spaces)} spaces.")
        if len(self._spaces) == 2:
            s0, s1 = self._spaces
            if s0.device != s1.device:
                raise ValueError("Spaces should have the same device, "
                                f"but got {s0.device} and {s1.device}.")
            if s0.ftype != s1.ftype:
                raise ValueError("Spaces should have the same dtype, "
                                f"but got {s0.ftype} and {s1.ftype}.")

    def _single_assembly(self, retain_ints: bool) -> Tensor:
        self.check_space()
        space = self._spaces
        device = space[0].device
        ugdof = space[0].number_of_global_dofs()
        vgdof = space[1].number_of_global_dofs() if (len(space) > 1) else ugdof
        global_mat_shape = (vgdof, ugdof)
        M = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=space[0].itype, device=device),
            torch.empty((0,), dtype=space[0].ftype, device=device),
            size=global_mat_shape
        )

        for group in self.integrators.keys():
            group_tensor, e2dofs = self._assembly_group(group, retain_ints)
            ue2dof = e2dofs[0]
            ve2dof = e2dofs[1] if (len(e2dofs) > 1) else ue2dof
            I = torch.broadcast_to(ve2dof[:, :, None], size=group_tensor.shape)
            J = torch.broadcast_to(ue2dof[:, None, :], size=group_tensor.shape)
            indices = torch.stack([I.ravel(), J.ravel()], dim=0)
            M += torch.sparse_coo_tensor(indices, group_tensor.ravel(), size=global_mat_shape)

        return M

    def _batch_assembly(self, retain_ints: bool, batch_size: int) -> Tensor:
        self.check_space()
        space = self._spaces
        device = space[0].device
        ugdof = space[0].number_of_global_dofs()
        vgdof = space[1].number_of_global_dofs() if (len(space) > 1) else ugdof
        ldof = space.number_of_local_dofs()
        global_mat_shape = (vgdof, ugdof, batch_size)
        M = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=space.itype, device=device),
            torch.empty((0, batch_size), dtype=space.ftype, device=device),
            size=global_mat_shape
        )

        for group in self.integrators.keys():
            group_tensor, e2dofs = self._assembly_group(group, retain_ints)
            ue2dof = e2dofs[0]
            ve2dof = e2dofs[1] if (len(e2dofs) > 1) else ue2dof
            NC = ue2dof.size(0)
            local_mat_shape = (batch_size, NC, ldof, ldof)

            if group_tensor.ndim == 3:
                group_tensor = group_tensor.unsqueeze(0).expand(local_mat_shape)

            I = torch.broadcast_to(ve2dof[:, :, None], size=group_tensor.shape)
            J = torch.broadcast_to(ue2dof[:, None, :], size=group_tensor.shape)
            indices = torch.stack([I.ravel(), J.ravel()], dim=0)
            group_tensor = group_tensor.reshape(batch_size, -1).transpose(0, 1)
            M += torch.sparse_coo_tensor(indices, group_tensor, size=global_mat_shape)

        return M

    def assembly(self, coalesce=True, retain_ints: bool=False) -> Tensor:
        """Assembly the bilinear form matrix.

        Parameters:
            coalesce (bool, optional): Whether to coalesce the sparse tensor.\n
            retain_ints (bool, optional): Whether to retain the integrator cache.

        Returns:
            Tensor[gdof, gdof]. Batch is placed in the LAST dimension as\
            Hybrid COO Tensor format.
        """
        if self.batch_size == 0:
            M = self._single_assembly(retain_ints)
        elif self.batch_size > 0:
            M = self._batch_assembly(retain_ints, self.batch_size)
        else:
            raise ValueError("batch_size must be a non-negative integer.")

        self._M = M.coalesce() if coalesce else M
        logger.info(f"Bilinear form matrix constructed, with shape {list(self._M.shape)}.")

        return self._M

    def mult(self, x: Tensor, out: Optional[Tensor]=None) -> Tensor:
        """Maxtrix vector multiplication.

        Parameters:
            x (Tensor): Vector, accepts batch on the first dimension.\n
            out (Tensor, optional): Output vector. Defaults to None.

        Returns:
            Tensor: self @ x
        """
        raise NotImplementedError

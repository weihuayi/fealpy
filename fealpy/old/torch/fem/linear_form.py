
from typing import Optional

import torch
from torch import Tensor

from .. import logger
from .form import Form


class LinearForm(Form):
    def check_local_shape(self, entity_to_global: Tensor, local_tensor: Tensor):
        if entity_to_global.ndim != 2:
            raise ValueError("entity-to-global relationship should be a 2D tensor, "
                             f"but got shape {tuple(entity_to_global.shape)}.")
        if entity_to_global.shape[0] != local_tensor.shape[0]:
            raise ValueError(f"entity_to_global.shape[0] != local_tensor.shape[0]")
        if local_tensor.ndim not in (2, 3):
            raise ValueError("Output of operator integrators should be 3D "
                             "(or 4D with batch in the first dimension), "
                             f"but got shape {tuple(local_tensor.shape)}.")

    def check_space(self):
        if len(self._spaces) != 1:
            raise ValueError("LinearForm should have only one space.")

    def _single_assembly(self, retain_ints: bool) -> Tensor:
        self.check_space()
        space = self._spaces[0]
        device = space.device
        gdof = space.number_of_global_dofs()
        global_mat_shape = (gdof,)
        M = torch.sparse_coo_tensor(
            torch.empty((1, 0), dtype=space.itype, device=device),
            torch.empty((0,), dtype=space.ftype, device=device),
            size=global_mat_shape
        )

        for group in self.integrators.keys():
            group_tensor, e2dofs = self._assembly_group(group, retain_ints)
            indices = e2dofs[0].reshape(1, -1)
            M += torch.sparse_coo_tensor(indices, group_tensor.ravel(), size=global_mat_shape)

        return M

    def _batch_assembly(self, retain_ints: bool, batch_size: int) -> Tensor:
        self.check_space()
        space = self._spaces[0]
        device = space.device
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()
        global_mat_shape = (gdof, batch_size)
        M = torch.sparse_coo_tensor(
            torch.empty((1, 0), dtype=space.itype, device=device),
            torch.empty((0, batch_size), dtype=space.ftype, device=device),
            size=global_mat_shape
        )

        for group in self.integrators.keys():
            group_tensor, e2dofs = self._assembly_group(group, retain_ints)
            NC = e2dofs[0].size(0)
            local_mat_shape = (batch_size, NC, ldof)

            if group_tensor.ndim == 2:
                group_tensor = group_tensor.unsqueeze(0).expand(local_mat_shape)

            indices = e2dofs[0].reshape(1, -1)
            group_tensor = group_tensor.reshape(batch_size, -1).transpose(0, 1)
            M += torch.sparse_coo_tensor(indices, group_tensor, size=global_mat_shape)

        return M

    def assembly(self, coalesce=True, retain_ints: bool=False, return_dense=True) -> Tensor:
        """Assembly the linear form vector.

        Parameters:
            coalesce (bool, optional): Whether to coalesce the sparse tensor.\n
            retain_ints (bool, optional): Whether to retain the integrator cache.\n
            return_dense (bool, optional): Whether to return dense tensor.

        Returns:
            Tensor[gdof,]. Batch is placed in the FIRST dimension for dense tensor,\
            and in the LAST dimension for sparse tensor (Hybrid COO Tensor).
        """
        if self.batch_size == 0:
            V = self._single_assembly(retain_ints)
        elif self.batch_size > 0:
            V = self._batch_assembly(retain_ints, self.batch_size)
        else:
            raise ValueError("batch_size must be a non-negative integer.")

        self._V = V.coalesce() if coalesce else V
        logger.info(f"Linear form vector constructed, with shape {list(V.shape)}.")

        if return_dense:
            if self.batch_size > 0:
                return self._V.to_dense().transpose_(0, 1)
            return self._V.to_dense()

        return self._V

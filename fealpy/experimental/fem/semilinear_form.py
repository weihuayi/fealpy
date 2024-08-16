
from typing import Optional

from ..typing import TensorLike
from ..backend import backend_manager as bm
from ..sparse import COOTensor
from .form import Form
from .. import logger


class SemilinearForm(Form):
    _M: Optional[COOTensor] = None
    _V: Optional[COOTensor] = None

    def _get_sparse_shape(self):
        pass

    def _scalar_assembly_A(self, retain_ints: bool, batch_size: int):

        space = self._spaces
        ugdof = space[0].number_of_global_dofs()
        vgdof = space[1].number_of_global_dofs() if (len(space) > 1) else ugdof
        init_value_shape = (0,) if (batch_size == 0) else (batch_size, 0)
        sparse_shape = (vgdof, ugdof)

        M = COOTensor(
            indices = bm.empty((2, 0), dtype=space[0].itype),
            values = bm.empty(init_value_shape, dtype=space[0].ftype),
            spshape = sparse_shape
        )

        for group in self.integrators.keys():
            group_tensor, e2dofs = self._assembly_group(group, retain_ints)
            ue2dof = e2dofs[0]
            ve2dof = e2dofs[1] if (len(e2dofs) > 1) else ue2dof
            local_shape = group_tensor.shape[-3:] # (NC, vldof, uldof)

            if (batch_size > 0) and (group_tensor.ndim == 3): # Case: no batch dimension
                group_tensor = bm.stack([group_tensor]*batch_size, axis=0)

            I = bm.broadcast_to(ve2dof[:, :, None], local_shape)
            J = bm.broadcast_to(ue2dof[:, None, :], local_shape)
            indices = bm.stack([I.ravel(), J.ravel()], axis=0)
            group_tensor = bm.reshape(group_tensor, self._values_ravel_shape)
            M = M.add(COOTensor(indices, group_tensor, sparse_shape))

        return M
    
    def _scalar_assembly_F(self, retain_ints: bool, batch_size: int):

        space = self._spaces[0]
        gdof = space.number_of_global_dofs()
        init_value_shape = (0,) if (batch_size == 0) else (batch_size, 0)
        sparse_shape = (gdof, )

        M = COOTensor(
            indices = bm.empty((1, 0), dtype=space.itype),
            values = bm.empty(init_value_shape, dtype=space.ftype),
            spshape = sparse_shape
        )

        for group in self.integrators.keys():
            group_tensor, e2dofs = self._assembly_group(group, retain_ints)

            if (batch_size > 0) and (group_tensor.ndim == 2):
                group_tensor = bm.stack([group_tensor]*batch_size, axis=0)

            indices = e2dofs[0].reshape(1, -1)
            group_tensor = bm.reshape(group_tensor, self._values_ravel_shape)
            M = M.add(COOTensor(indices, group_tensor, sparse_shape))

        return M
    
    def assembly(self, *, return_dense=True, coalesce=True, retain_ints: bool=False) -> COOTensor:
        """Assembly the bilinear form matrix.

        Parameters:
            coalesce (bool, optional): Whether to coalesce the sparse tensor.\n
            retain_ints (bool, optional): Whether to retain the integrator cache.

        Returns:
            global_matrix (COOTensor): Global sparse matrix shaped ([batch, ]gdof, gdof).
        """
        M = self._scalar_assembly_A(retain_ints, self.batch_size)

        self._M = M.coalesce() if coalesce else M

        V = self._scalar_assembly_F(retain_ints, self.batch_size)

        self._V = V.coalesce() if coalesce else V
        if return_dense:
            return self._M, self._V.to_dense()

        return self._M, self._V
    
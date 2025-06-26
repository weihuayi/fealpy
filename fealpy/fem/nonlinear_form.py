from typing import Optional

from ..backend import backend_manager as bm
from ..sparse import COOTensor
from .form import Form
from .integrator import NonlinearInt, OpInt, SrcInt
from .nonlinear_wrapper import NonlinearWrapperInt


class NonlinearForm(Form[NonlinearInt]):
    _M: Optional[COOTensor] = None
    _V: Optional[COOTensor] = None

    def _get_sparse_shape(self):
        spaces = self._spaces
        ugdof = spaces[0].number_of_global_dofs()
        vgdof = spaces[1].number_of_global_dofs() if (len(spaces) > 1) else ugdof
        return (vgdof, ugdof)

    def _assembly_kernel(self, group: str, /, indices=None):
        integrator = self.integrators[group]
        if indices is None:
            if isinstance(integrator, OpInt) and not isinstance(integrator, NonlinearInt):
                integrator = NonlinearWrapperInt(integrator)
            value = integrator.assembly(self.space)
            etg = integrator.to_global_dof(self.space)
        else:
            value = integrator.assembly(self.space, indices=indices)
            etg = integrator.to_global_dof(self.space, indices=indices)
        if not isinstance(etg, (tuple, list)):
            etg = (etg, )
        return value, etg

    def _scaler_assembly(self):
        space = self._spaces
        batch_size = self.batch_size
        ugdof = space[0].number_of_global_dofs()
        vgdof = space[1].number_of_global_dofs() if (len(space) > 1) else ugdof
        init_value_shape = (0,) if (batch_size == 0) else (batch_size, 0)
        device = space[0].device

        M = COOTensor(
            indices = bm.empty((2, 0), dtype=space[0].itype, device=device),
            values = bm.empty(init_value_shape, dtype=space[0].ftype, device=device),
            spshape = (vgdof, ugdof)
        )
        V = COOTensor(
            indices = bm.empty((1, 0), dtype=space[0].itype, device=device),
            values = bm.empty(init_value_shape, dtype=space[0].ftype, device=device),
            spshape = (ugdof, )
        )
        for group_tensor, e2dofs_tuple in self.assembly_local_iterative():
            if isinstance(group_tensor, tuple):
                op_tensor, src_tensor = group_tensor
            else:
                op_tensor, src_tensor = None, group_tensor

            if op_tensor is not None:
                ue2dof = e2dofs_tuple[0]
                ve2dof = e2dofs_tuple[1] if (len(e2dofs_tuple) > 1) else ue2dof
                local_shape = op_tensor.shape[-3:] # (NC, vldof, uldof)

                if (batch_size > 0) and (op_tensor.ndim == 3): # Case: no batch dimension
                    op_tensor = bm.stack([op_tensor]*batch_size, axis=0)
                I = bm.broadcast_to(ve2dof[:, :, None], local_shape)
                J = bm.broadcast_to(ue2dof[:, None, :], local_shape)
                indices = bm.stack([I.ravel(), J.ravel()], axis=0)
                op_tensor = bm.reshape(op_tensor, self._values_ravel_shape)
                M = M.add(COOTensor(indices, op_tensor, (vgdof, ugdof)))

            if (batch_size > 0) and (src_tensor.ndim == 2):
                src_tensor = bm.stack([src_tensor]*batch_size, axis=0)
            indices = e2dofs_tuple[0].reshape(1, -1)
            src_tensor = bm.reshape(src_tensor, self._values_ravel_shape)
            V = V.add(COOTensor(indices, src_tensor, (ugdof, )))

        return M, V

    def assembly(self, *, return_dense=True, coalesce=True, format='csr') -> COOTensor:

        # M = self._scalar_assembly_A(retain_ints, self.batch_size)
        M, V = self._scaler_assembly()

        if format == 'csr':
            self._M = M.coalesce().tocsr()
        elif format == 'coo':
            self._M = M.coalesce()
        else:
            raise ValueError(f"Unsupported format {format}.")

        # V = self._scalar_assembly_F(retain_ints, self.batch_size)
        self._V = V.coalesce() if coalesce else V
        if return_dense:
            return self._M, self._V.to_dense()

        return self._M, self._V
    
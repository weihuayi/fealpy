from typing import Optional

from .. import logger
from ..typing import TensorLike
from ..backend import backend_manager as bm
from ..sparse import COOTensor
from .form import Form
from .integrator import NonlinearInt, OpInt, SrcInt
from .nonlinear_wrapper import NonlinearWrapperInt


class NonlinearForm(Form[NonlinearInt]):
    _M: Optional[COOTensor] = None
    _V: Optional[COOTensor] = None

    def _get_sparse_shape(self):
        pass

    def _assembly_group(self, group: str, retain_ints: bool=False):
        if group in self.memory:
            return self.memory[group]

        INTS = self.integrators[group]
        SEMILINEAR_INTS = []
        etg = [INTS[0].to_global_dof(s) for s in self._spaces]

        for ints_ in INTS:
            if isinstance(ints_, OpInt):
                if not isinstance(ints_, NonlinearInt):
                    ints_ = NonlinearWrapperInt(ints_)
                SEMILINEAR_INTS.append(ints_)

        ct_F = SEMILINEAR_INTS[0](self.space)[1] if len(SEMILINEAR_INTS) !=0 else None

        if isinstance(INTS[0], OpInt):
            if not isinstance(INTS[0], NonlinearInt):
                ct_A = NonlinearWrapperInt(INTS[0])(self.space)[0]
            else:
                ct_A = INTS[0](self.space)[0]
        else:
            ct_A = INTS[0](self.space)

        for int_ in INTS[1:]:
            new_ct_A = int_(self.space)[0]
            fdim_A = min(ct_A.ndim, new_ct_A.ndim)
            if ct_A.shape[:fdim_A] != new_ct_A.shape[:fdim_A]:
                raise RuntimeError(f"The output of the integrator {int_.__class__.__name__} "
                                f"has an incompatible shape {tuple(new_ct_A.shape)} "
                                f"with the previous {tuple(ct_A.shape)} in the group '{group}'.")
            if new_ct_A.ndim > ct_A.ndim:
                ct_A = new_ct_A + ct_A[None, ...]
            elif new_ct_A.ndim < ct_A.ndim:
                ct_A = ct_A + new_ct_A[None, ...]
            else:
                ct_A = ct_A+ new_ct_A

        for int_sem_ in SEMILINEAR_INTS[1:]:
            new_ct_F = int_sem_(self.space)[1]
            fdim_F = min(ct_F.ndim, new_ct_F.ndim)
            if ct_F.shape[:fdim_F] != new_ct_F.shape[:fdim_F]:
                raise RuntimeError(f"The output of the integrator {int_.__class__.__name__} "
                                f"has an incompatible shape {tuple(new_ct_F.shape)} "
                                f"with the previous {tuple(ct_F.shape)} in the group '{group}'.")

            if new_ct_F.ndim > ct_F.ndim:
                ct_F = new_ct_F + ct_F[None, ...]
            elif new_ct_F.ndim < ct_F.ndim:
                ct_F = ct_F + new_ct_F[None, ...]
            else:
                ct_F = ct_F+ new_ct_F

            if retain_ints:
                self.memory[group] = ((ct_A, etg), (ct_F, etg))

        return (ct_A, etg), (ct_F, etg)

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
            if isinstance(self.integrators[group][0], OpInt):
                group_tensor, e2dofs = self._assembly_group(group, retain_ints)[0]
                ue2dof = e2dofs[0]
                ve2dof = e2dofs[1] if (len(e2dofs) > 1) else ue2dof
                local_shape = group_tensor.shape[-3:] # (NC, vldof, uldof)

                if (batch_size > 0) and (group_tensor.ndim == 3): # Case: no batch dimension
                    group_tensor = bm.stack([group_tensor]*batch_size, axis=0)
                J = bm.broadcast_to(ue2dof[:, None, :], local_shape)
                I = bm.broadcast_to(ve2dof[:, :, None], local_shape)
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
            if isinstance(self.integrators[group][0], SrcInt):
                group_tensor, e2dofs = self._assembly_group(group, retain_ints)[0]
            else:
                group_tensor, e2dofs = self._assembly_group(group, retain_ints)[1]

            if (batch_size > 0) and (group_tensor.ndim == 2):
                group_tensor = bm.stack([group_tensor]*batch_size, axis=0)

            indices = e2dofs[0].reshape(1, -1)
            group_tensor = bm.reshape(group_tensor, self._values_ravel_shape)
            M = M.add(COOTensor(indices, group_tensor, sparse_shape))

        return M

    def assembly(self, *, return_dense=True, coalesce=True, format='csr',retain_ints: bool=False) -> COOTensor:

        M = self._scalar_assembly_A(retain_ints, self.batch_size)
        if format == 'csr':
            self._M = M.coalesce().tocsr()
        elif format == 'coo':
            self._M = M.coalesce()
        else:
            raise ValueError(f"Unsupported format {format}.")

        V = self._scalar_assembly_F(retain_ints, self.batch_size)
        self._V = V.coalesce() if coalesce else V
        if return_dense:
            return self._M, self._V.to_dense()

        return self._M, self._V
    
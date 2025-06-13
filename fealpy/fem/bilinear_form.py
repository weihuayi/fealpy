
from typing import Optional, Literal, overload

from .. import logger
from ..typing import TensorLike
from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor
from .form import Form
from .integrator import LinearInt


class BilinearForm(Form[LinearInt]):
    _M = None

    def _get_sparse_shape(self):
        spaces = self._spaces
        ugdof = spaces[0].number_of_global_dofs()
        vgdof = spaces[1].number_of_global_dofs() if (len(spaces) > 1) else ugdof
        return (vgdof, ugdof)

    def check_local_shape(self, entity_to_global: TensorLike, local_tensor:
                          TensorLike):
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
            if s0.mesh.device != s1.mesh.device:
                raise ValueError("Spaces should have the same device, "
                                f"but got {s0.device} and {s1.device}.")
            if s0.ftype != s1.ftype:
                raise ValueError("Spaces should have the same dtype, "
                                f"but got {s0.ftype} and {s1.ftype}.")

    def _scalar_assembly(self):
        self.check_space()
        space = self._spaces
        batch_size = self.batch_size
        ugdof = space[0].number_of_global_dofs()
        vgdof = space[1].number_of_global_dofs() if (len(space) > 1) else ugdof
        init_value_shape = (0,) if (batch_size == 0) else (batch_size, 0)
        sparse_shape = (vgdof, ugdof)

        M = COOTensor(
            indices = bm.empty((2, 0), dtype=space[0].itype, device=bm.get_device(space[0])),
            values = bm.empty(init_value_shape, dtype=space[0].ftype, device=bm.get_device(space[0])),
            spshape = sparse_shape
        )
        # for group in self.integrators.keys():
            # group_tensor, e2dofs = self._assembly_group(group, retain_ints)
        for group_tensor, e2dofs_tuple in self.assembly_local_iterative():
            ue2dof = e2dofs_tuple[0]
            ve2dof = e2dofs_tuple[1] if (len(e2dofs_tuple) > 1) else ue2dof
            local_shape = group_tensor.shape[-3:] # (NC, vldof, uldof)

            if (batch_size > 0) and (group_tensor.ndim == 3): # Case: no batch dimension
                group_tensor = bm.stack([group_tensor]*batch_size, axis=0)
            I = bm.broadcast_to(ve2dof[:, :, None], local_shape)
            J = bm.broadcast_to(ue2dof[:, None, :], local_shape)
            indices = bm.stack([I.ravel(), J.ravel()], axis=0)
            group_tensor = bm.reshape(group_tensor, self._values_ravel_shape)
            M = M.add(COOTensor(indices, group_tensor, sparse_shape))

        return M

    @overload
    def assembly(self) -> CSRTensor: ...
    @overload
    def assembly(self, *, format: Literal['coo']) -> COOTensor: ...
    @overload
    def assembly(self, *, format: Literal['csr']) -> CSRTensor: ...
    def assembly(self, *, format='csr'):
        """Assembly the bilinear form matrix.

        Parameters:
            format (str, optional): Layout of the output ('csr' | 'coo'). Defaults to 'csr'.\n
            retain_ints (bool, optional): Whether to retain the integrator cache.csr

        Returns:
            global_matrix (CSRTensor | COOTensor): Global sparse matrix shaped ([batch, ]gdof, gdof).
        """
        M = self._scalar_assembly()
        if getattr(self, '_transposed', False):
            M = M.T

        if format == 'csr':
            self._M = M.coalesce().tocsr()
        elif format == 'coo':
            self._M = M.coalesce()
        else:
            raise ValueError(f"Unsupported format {format}.")
        logger.info(f"Bilinear form matrix constructed, with shape {list(self._M.shape)}.")

        return self._M

    def mult(self, x: TensorLike, out: Optional[TensorLike]=None) -> TensorLike:
        """Maxtrix vector multiplication.

        Parameters:
            x (TensorLike): Vector, accepts batch on the first dimension.\n
            out (TensorLike, optional): Output vector. Defaults to None.

        Returns:
            TensorLike: self @ x
        """
        raise NotImplementedError

    @property
    def T(self):
        transposed = self.copy()
        transposed._transposed = True
        transposed._M = self._M
        return transposed

    def __matmul__(self, u: TensorLike):
        if self._M is not None:
            return self._M @ u

        nrow = self.shape[-2]
        kwargs = bm.context(u)

        if self.batch_size > 0:
            shape = (self.batch_size, nrow)
            out_subs = 'bci'
            gv_reshape = (self.batch_size, -1)
        else:
            if u.ndim >= 2:
                shape = (u.shape[0], nrow)
                out_subs = 'bci'
                gv_reshape = (u.shape[0], -1)
            else:
                shape = (nrow,)
                out_subs = 'ci'
                gv_reshape = (-1,)

        v = bm.zeros(shape, **kwargs)
        gt_subs = 'bcij' if (self.batch_size > 0) else 'cij'
        gu_subs = 'bcj' if (u.ndim >= 2) else 'cj'

        for group_tensor, e2dofs_tuple in self.assembly_local_iterative():
            ue2dof = e2dofs_tuple[0]
            ve2dof = e2dofs_tuple[1] if (len(e2dofs_tuple) > 1) else ue2dof
            gu = u[..., ue2dof] # (..., NC, uldof)
            gv = bm.einsum(f'{gt_subs}, {gu_subs} -> {out_subs}', group_tensor, gu)
            v = bm.index_add(v, ve2dof.reshape(-1), gv.reshape(gv_reshape))

        return v



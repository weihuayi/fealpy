
from typing import Optional

from ..typing import TensorLike
from ..backend import backend_manager as bm
from ..sparse import COOTensor
from .form import Form
from .. import logger


class BilinearForm(Form):
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
            if s0.device != s1.device:
                raise ValueError("Spaces should have the same device, "
                                f"but got {s0.device} and {s1.device}.")
            if s0.ftype != s1.ftype:
                raise ValueError("Spaces should have the same dtype, "
                                f"but got {s0.ftype} and {s1.ftype}.")

    def _scalar_assembly(self, retain_ints: bool, batch_size: int):
        self.check_space()
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
            local_shape = group_tensor.shape[-3:] # (NC, ldof, ldof)

            if (batch_size > 0) and (group_tensor.ndim == 3): # Case: no batch dimension
                group_tensor = bm.stack([group_tensor]*batch_size, axis=0)

            I = bm.broadcast_to(ve2dof[:, :, None], local_shape)
            J = bm.broadcast_to(ue2dof[:, None, :], local_shape)
            indices = bm.stack([I.ravel(), J.ravel()], axis=0)
            group_tensor = bm.reshape(group_tensor, self._values_ravel_shape)
            M = M.add(COOTensor(indices, group_tensor, sparse_shape))

        return M

    def assembly(self, *, coalesce=True, retain_ints: bool=False) -> COOTensor:
        """Assembly the bilinear form matrix.

        Parameters:
            coalesce (bool, optional): Whether to coalesce the sparse tensor.\n
            retain_ints (bool, optional): Whether to retain the integrator cache.

        Returns:
            global_matrix (COOTensor): Global sparse matrix shaped ([batch, ]gdof, gdof).
        """
        M = self._scalar_assembly(retain_ints, self.batch_size)

        self._M = M.coalesce() if coalesce else M
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

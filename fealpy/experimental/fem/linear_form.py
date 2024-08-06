
from typing import overload, Literal

from ..typing import TensorLike
from ..backend import backend_manager as bm 
from ..sparse import COOTensor
from .form import Form
from .. import logger


class LinearForm(Form):
    def check_local_shape(self, entity_to_global: TensorLike, local_tensor:
                          TensorLike):
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

    def _scalar_assembly(self, retain_ints: bool, batch_size: int):
        self.check_space()
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

    @overload
    def assembly(self, *, return_dense: Literal[True]=True, coalesce=True, retain_ints: bool=False) -> TensorLike: ...
    @overload
    def assembly(self, *, return_dense: Literal[False], coalesce=True, retain_ints: bool=False) -> COOTensor: ...
    def assembly(self, *, return_dense=True, coalesce=True, retain_ints: bool=False):
        """Assembly the linear form vector.

        Parameters:
            return_dense (bool, optional): Whether to return dense tensor.\n
            coalesce (bool, optional): Whether to coalesce the sparse tensor.\n
            retain_ints (bool, optional): Whether to retain the integrator cache.

        Returns:
            global_vector (COOTensor | TensorLike): Global sparse vector shaped ([batch, ]gdof).
        """
        V = self._scalar_assembly(retain_ints, self.batch_size)

        self._V = V.coalesce() if coalesce else V
        logger.info(f"Linear form vector constructed, with shape {list(V.shape)}.")

        if return_dense:
            return self._V.to_dense()

        return self._V

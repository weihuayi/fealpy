
from typing import overload, Literal, Optional

from .. import logger
from ..typing import TensorLike
from ..backend import backend_manager as bm 
from ..sparse import COOTensor
from .form import Form
from .integrator import LinearInt


class LinearForm(Form[LinearInt]):
    _V = None

    def _get_sparse_shape(self):
        spaces = self._spaces
        ugdof = spaces[0].number_of_global_dofs()
        return (ugdof,)

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

    def _scalar_assembly(self):
        self.check_space()
        space = self._spaces[0]
        batch_size = self.batch_size
        gdof = space.number_of_global_dofs()
        init_value_shape = (0,) if (batch_size == 0) else (batch_size, 0)
        sparse_shape = (gdof, )

        M = COOTensor(
            indices = bm.empty((1, 0), dtype=space.itype, device=bm.get_device(space)),
            values = bm.empty(init_value_shape, dtype=space.ftype, device=bm.get_device(space)),
            spshape = sparse_shape
        )

        for group_tensor, e2dofs_tuple in self.assembly_local_iterative():
            if (batch_size > 0) and (group_tensor.ndim == 2):
                group_tensor = bm.stack([group_tensor]*batch_size, axis=0)

            indices = e2dofs_tuple[0].reshape(1, -1)
            group_tensor = bm.reshape(group_tensor, self._values_ravel_shape)
            M = M.add(COOTensor(indices, group_tensor, sparse_shape))

        return M

    @overload
    def assembly(self) -> TensorLike: ...
    @overload
    def assembly(self, *, format: Literal['coo']) -> COOTensor: ...
    @overload
    def assembly(self, *, format: Literal['dense']) -> TensorLike: ...
    def assembly(self, *, format='dense'):
        """Assembly the linear form vector.

        Parameters:
            format (str, optional): Layout of the output ('dense', 'coo'). Defaults to 'dense'.\n
            retain_ints (bool, optional): Whether to retain the integrator cache.

        Returns:
            global_vector (COOTensor | TensorLike): Global sparse vector shaped ([batch, ]gdof).
        """
        V = self._scalar_assembly()

        if format == 'dense':
            self._V = V.to_dense()
        elif format == 'coo':
            self._V = V.coalesce()
        else:
            raise ValueError(f"Unsupported format {format}.")
        logger.info(f"Linear form vector constructed, with shape {list(V.shape)}.")

        return self._V

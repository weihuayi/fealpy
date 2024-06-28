
from typing import TypeVar, Optional

import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR

from .. import logger
from ..functionspace.space import FunctionSpace
from ..mesh.utils import Array
from .form import Form


_FS = TypeVar('_FS', bound=FunctionSpace)


class LinearForm(Form[_FS]):
    def check_local_shape(self, entity_to_global: Array, local_tensor: Array):
        if entity_to_global.ndim != 2:
            raise ValueError("entity-to-global relationship should be a 2D tensor, "
                             f"but got shape {tuple(entity_to_global.shape)}.")
        if entity_to_global.shape[0] != local_tensor.shape[0]:
            raise ValueError(f"entity_to_global.shape[0] != local_tensor.shape[0]")
        if local_tensor.ndim not in (2, 3):
            raise ValueError("Output of operator integrators should be 3D "
                             "(or 4D with batch in the first dimension), "
                             f"but got shape {tuple(local_tensor.shape)}.")
        
    def assembly(self, retain_ints: bool=False) -> Array:
        
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间组成的向量函数空间
            return self.assembly_for_vspace_with_scalar_basis()
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.assembly_for_sspace_and_vspace_with_vector_basis(retain_ints=retain_ints)
        
    def assembly_for_sspace_and_vspace_with_vector_basis(self, retain_ints: bool) -> BCOO:
        space = self.space
        gdof = space.number_of_global_dofs()
        global_mat_shape = (gdof, )
        # M = BCOO._empty(shape=global_mat_shape, dtype=space.ftype, index_dtype=space.itype)
        self._V = jnp.zeros(global_mat_shape, dtype=space.ftype)
        
        for group in self.integrators.keys():
            group_tensor, e2dof = self._assembly_group(group, retain_ints)
            # indices = e2dof.reshape(1, -1).T
            self._V = self._V.at[e2dof.reshape(-1, ).T].add(group_tensor.ravel())
            # M += BCOO((group_tensor.ravel(), indices), shape=global_mat_shape)
        return self._V

    def assembly_for_vspace_with_scalar_basis(self, retain_ints: bool):
        pass

from typing import TypeVar, Optional

import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from .. import logger
from ..functionspace.space import FunctionSpace
from ..mesh.utils import Array
from .form import Form


_FS = TypeVar('_FS', bound=FunctionSpace)


class BilinearForm(Form[_FS]):
    def check_local_shape(self, entity_to_global: Array, local_tensor: Array):
        if entity_to_global.ndim != 2:
            raise ValueError("entity-to-global relationship should be a 2D tensor, "
                             f"but got shape {tuple(entity_to_global.shape)}.")
        if entity_to_global.shape[0] != local_tensor.shape[0]:
            raise ValueError(f"entity_to_global.shape[0] != local_tensor.shape[0]")
        if local_tensor.ndim not in (3, 4):
            raise ValueError("Output of operator integrators should be 3D "
                             "(or 4D with batch in the first dimension), "
                             f"but got shape {tuple(local_tensor.shape)}.")

    def assembly(self):
        """
        @brief 数值积分组装

        @note space 可能是以下的情形
            * 标量空间
            * 由标量空间组成的向量空间
            * 由标量空间组成的张量空间
            * 向量空间（基函数是向量型的）
            * 张量空间（基函数是张量型的
        """
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间组成的向量函数空间
            return self.assembly_for_vspace_with_scalar_basis()
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.assembly_for_sspace_and_vspace_with_vector_basis(retain_ints=False)
    
    def assembly_for_sspace_and_vspace_with_vector_basis(self, retain_ints: bool) ->BCOO:
        
        space = self.space
        gdof = space.number_of_global_dofs()
        global_mat_shape = (gdof, gdof)
        
        M = BCOO._empty(shape=global_mat_shape, dtype=space.ftype, index_dtype=space.itype)
        
        for group in self.integrators.keys():
            group_tensor, e2dof = self._assembly_group(group, retain_ints)
            # Broadcast and flatten indices I and J
            I = jnp.broadcast_to(e2dof[:, :, None], shape=group_tensor.shape)
            J = jnp.broadcast_to(e2dof[:, None, :], shape=group_tensor.shape)
            I = I.ravel()
            J = J.ravel()
            indices = jnp.stack([I, J], axis=0).T
            M += BCOO((group_tensor.ravel(), indices), shape=global_mat_shape)
        self.M = M
        
        return self.M
    
    def assembly_for_vspace_with_scalar_basis(self, retain_ints: bool) -> BCOO:
        pass

    def mult(self, x: Array, out: Optional[Array]=None) -> Array:
        """Maxtrix vector multiplication.

        Args:
            x (Tensor): Vector, accepts batch on the first dimension.
            out (Tensor, optional): Output vector. Defaults to None.

        Returns:
            Tensor: self @ x
        """
        raise NotImplementedError

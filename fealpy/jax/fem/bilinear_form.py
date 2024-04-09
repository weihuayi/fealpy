
import numpy as np
from scipy.sparse import csr_matrix

import jax 
import jax.numpy as jnp

from .. import logger

class BilinearForm:
    """
    @brief 试探函数和测试函数空间相同的双线性型
    """
    def __init__(self, space, atype=None):
        """
        @brief 
        """
        self.space = space
        self.dintegrators = [] # 区域积分子
        self.bintegrators = [] # 边界积分子

        self._M = None # 需要组装的矩阵 

    def add_domain_integrator(self, I) -> None:
        """
        @brief 增加一个或多个区域积分对象
        """
        if isinstance(I, list):
            self.dintegrators.extend(I)
        else:
            self.dintegrators.append(I)


    def add_boundary_integrator(self, I):
        """
        @brief 增加一个或多个边界积分对象
        """
        if isinstance(I, list):
            self.bintegrators.extend(I)
        else:
            self.bintegrators.append(I)

    def assembly(self):
        """
        @brief 数值积分组装
        """
        space = self.space
        ldof = space.number_of_local_dofs()
        gdof = space.number_of_global_dofs()

        mesh = space.mesh
        NC = mesh.number_of_cells()
        CM = self.dintegrators[0].assembly_cell_matrix(space) 
        for di in self.dintegrators[1:]:
            CM = CM + di.assembly_cell_matrix(space)

        cell2dof = space.cell_to_dof()
        I = jnp.broadcast_to(cell2dof[:, :, None], shape=CM.shape)
        J = jnp.broadcast_to(cell2dof[:, None, :], shape=CM.shape)
        self._M = csr_matrix((CM.ravel(), (I.ravel(), J.ravel())), shape=(gdof, gdof))

        logger.info(f"Finished construct bilinear from matrix with shape {self._M.shape}.")
        return self._M

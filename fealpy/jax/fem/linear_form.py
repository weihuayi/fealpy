
import numpy as np

import jax
import jax.numpy as jnp

class LinearForm:
    """

    """
    def __init__(self, space):
        """
        @brief 
        """
        self.space = space
        self._V = None # 需要组装的向量
        self.dintegrators = [] # 区域积分子
        self.bintegrators = [] # 边界积分子

    def add_domain_integrator(self, I):
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
        mesh = space.mesh

        NC = mesh.number_of_cells()
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()

        bb = np.zeros((NC, ldof), dtype=space.ftype)
        bb = self.dintegrators[0].assembly_cell_vector(space)
        for di in self.dintegrators[1:]:
            bb = bb + di.assembly_cell_vector(space)

        cell2dof = space.cell_to_dof()
        V = jnp.zeros((gdof, ), dtype=space.ftype)
        self._V = V.at[cell2dof].add(bb)
        return self._V

    def update(self):
        """
        @brief 当空间改变时，重新组装向量
        """
        return self.assembly()


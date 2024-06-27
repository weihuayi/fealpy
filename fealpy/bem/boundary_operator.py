import numpy as np
from typing import Optional, Union, Tuple, Callable, Any

class BoundaryOperator:

    def __init__(self, space):
        self.space = space
        self._H = None
        self._G = None
        self.dintegrators = []  # 区域积分子
        self.bintegrators = []  # 边界积分子


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

        @note space 可能是以下的情形, 程序上需要更好的设计
            * 标量空间
            * 由标量空间组成的向量空间
            * 由标量空间组成的张量空间
            * 向量空间（基函数是向量型的）
            * 张量空间（基函数是张量型的
        """
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间张成的向量函数空间
            return self.assembly_for_vspace_with_scalar_basis()
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.assembly_for_sspace_and_vspace_with_vector_basis()

    def assembly_for_sspace_and_vspace_with_vector_basis(self):
        # ===================================================
        space = self.space
        if space.p == 0:
            gdof = space.mesh.number_of_cells()
        else:
            gdof = space.dof.number_of_global_dofs()

        Hij, Gij = self.bintegrators[0].assembly_face_matrix(space)

        face2dof = space.dof.cell_to_dof()
        I = np.broadcast_to(np.arange(gdof, dtype=np.int64)[:, None, None], shape=Hij.shape)
        J = np.broadcast_to(face2dof[None, ...], shape=Hij.shape)

        # 整体矩阵的初始化与组装
        self._H = np.zeros((gdof, gdof))
        np.add.at(self._H, (I, J), Hij)
        np.fill_diagonal(self._H, 0.5)
        self._G = np.zeros((gdof, gdof))
        np.add.at(self._G, (I, J), Gij)
        bd_face_measure = space.mesh.entity_measure('cell')
        # TODO: 补充高次与高维情况下，对奇异积分的处理
        if space.GD == 2:
            np.fill_diagonal(self._G, (bd_face_measure * (np.log(2 / bd_face_measure) + 1) / np.pi / 2))
        # ===================================================
        f = self.dintegrators[0].assembly_cell_vector(space)
        self._f = f

        return self._H, self._G, self._f

    def assembly_for_vspace_with_scalar_basis(self):

        raise NotImplementedError

    def update(self):
        """
        @brief 当空间改变时，重新组装向量
        """
        return self.assembly()

import numpy as np
from scipy.sparse import csr_matrix


class InternalOperator:

    def __init__(self, space):
        self.space = space
        self._H = None
        self._G = None
        self.dintegrators = []  # 区域积分子
        self.bintegrators = []  # 边界积分子
        # 构造边界网格与空间
        self.bd_space = self.space.bd_space


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
        bd_space = self.bd_space

        gdof = self.space.number_of_global_dofs()

        Hij, Gij = self.bintegrators[0].assembly_face_matrix(bd_space, self.space)

        face2dof = self.space.bd_space.cell_to_dof()
        I = np.broadcast_to(np.arange(gdof, dtype=np.int64)[:, None, None], shape=Hij.shape)
        J = np.broadcast_to(face2dof[None, ...], shape=Hij.shape)

        # 整体矩阵的初始化与组装
        index = self.space.is_boundary_dof()
        self._H = np.zeros((gdof, gdof))
        np.add.at(self._H[index, index], (I, J), Hij)
        np.fill_diagonal(self._H, 0.5)
        self._G = np.zeros((gdof, gdof))
        np.add.at(self._G[index, index], (I, J), Gij)
        # ===================================================
        cell_space = self.space
        cell_gdof = cell_space.dof.number_of_global_dofs()

        f = self.dintegrators[0].assembly_cell_vector(cell_space, cell_space)
        self._f[index] = f

        return self._H, self._G, self._f

    def assembly_for_vspace_with_scalar_basis(self):

        raise NotImplementedError

    def update(self):
        """
        @brief 当空间改变时，重新组装向量
        """
        return self.assembly()


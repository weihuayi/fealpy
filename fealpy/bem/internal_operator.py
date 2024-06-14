import numpy as np
from scipy.sparse import csr_matrix


class InternalOperator:

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
        domain_mesh = space.domain_mesh
        xi = domain_mesh.entity('node')[~domain_mesh.ds.boundary_node_flag()]

        Hij, Gij = self.bintegrators[0].assembly_face_matrix(space, xi)

        face2dof = space.cell_to_dof()
        I = np.broadcast_to(np.arange(len(xi), dtype=np.int64)[:, None, None], shape=Hij.shape)
        J = np.broadcast_to(face2dof[None, ...], shape=Hij.shape)

        # 整体矩阵的初始化与组装
        self._H = np.zeros((len(xi), gdof))
        np.add.at(self._H, (I, J), Hij)
        self._G = np.zeros((len(xi), gdof))
        np.add.at(self._G, (I, J), Gij)
        # ===================================================
        f = self.dintegrators[0].assembly_cell_vector(space, xi)
        self._f = f

        return self._H, self._G, self._f

    def assembly_for_vspace_with_scalar_basis(self):

        raise NotImplementedError

    def update(self):
        """
        @brief 当空间改变时，重新组装向量
        """
        return self.assembly()


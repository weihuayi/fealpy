import numpy as np
from scipy.sparse import csr_matrix

class LinearForm:
    """

    """
    def __init__(self, space, atype=None):
        """
        @brief 
        """
        self.space = space
        self._V = None # 需要组装的矩阵 
        self.atype = atype # 矩阵组装的方式，None、fast、ref
        self.dintegrators = [] # 区域积分子
        self.bintegrators = [] # 边界积分子

    def add_domain_integrator(self, I):
        """
        @brief 增加一个区域积分对象
        """
        self.dintegrators.append(I)


    def add_boundary_integrator(self, I):
        """
        @brief 增加一个边界积分对象
        """
        self.bintegrators.append(I)

    def get_vector(self, copy=False):
        """
        @brief 获取线性型组装的向量

        @note 在关于时间的问题中
        """
        if copy is False:
            return self._V 
        else:
            return self._V.copy()

    def assembly(self):
        """
        @brief 数值积分组装

        @note space 可能是以下的情形, 程序上需要更好的设计
            * 标量空间
            * 由标量空间组成的向量空间
            * 由标量空间组成的张量空间
            * 向量空间（基函数是向量型的）
            * 张量空间（基函数是张量型的
        注意！目前仅实现的标量空间的情形
        """
        space = self.space
        mesh = space.mesh

        NC = mesh.number_of_cells()
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()

        bb = np.zeros((NC, ldof), dtype=space.ftype)
        for inte in self.dintegrators:
            inte.assembly_cell_vector(space, out=bb)

        cell2dof = space.cell_to_dof()
        self._V = np.zeros((gdof, ), dtype=space.ftype)
        np.add.at(self._V, cell2dof, bb)

    def update(self):
        """
        @brief 当空间改变时，重新组装向量
        """
        pass


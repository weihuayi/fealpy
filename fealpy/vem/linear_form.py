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

    def assembly(self, q = None):
        """
        @brief 数值积分组装
        """
        space = self.space
        bb = self.dintegrators[0].assembly_cell_vector(space, q=q)
        for i in range(len(self.dintegrators))[1:]:
            bb += self.dintegrators[i].assembly_cell_vector(space, q = q)
        gdof = space.number_of_global_dofs()
        self._V = np.bincount(np.concatenate(space.dof.cell2dof), weights=bb, minlength=gdof)
        return self._V

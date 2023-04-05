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
        self.M = None # 需要组装的矩阵 
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

    def assembly(self):
        """
        @brief 数值积分组装
        """
        space = self.space

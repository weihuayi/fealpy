import numpy as np

class BilinearForm:
    """
    """
    def __init__(self, space0, space1=None， atype=None):
        self.space0 = space0
        self.space1 = space0 if space1 is None else space1
        self.M = None # 需要组装的矩阵 
        self.atype = atype # 矩阵组装的方式，None、fast、ref
        self.dintegrators = [] # 区域积分子
        self.bintegrators = [] # 边界积分子

    def add_domain_integrator(self, I):
        """
        @brief 增加一个区域积分对象
        """
        self.dints.append(I)


    def add_boundary_integrator(self, I):
        """
        @brief 增加一个边界积分对象
        """
        self.bints.append(I)

    def mult(self, x, out=None):
        """
        """
        if out is None:
            return self.M@x
        else:
            out[:] = self.M@x

    def add_mult(self, x, y, a=1.0):
        y += a*(self.M@x)


    def assembly(self):
        """
        @brief 调用积分子组装矩阵
        """

    def fast_assembly(self):
        """
        @brief 免数值积分组装
        """

    def parallel_assembly(self):
        """
        @brief 多线程数值积分组装
        @note 特别当三维情形，
        """



import numpy as np

from .Operator import Operator


class BilinearForm(Operator):
    """
    """
    def __init__(self, space):
        self.space = space
        self.M = None # 
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



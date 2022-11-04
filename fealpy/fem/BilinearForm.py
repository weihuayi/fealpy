import numpy as np

from .Operator import Operator


class BilinearForm(Operator):
    """
    """
    def __init__(self, space):
        self.space = space
        self.M = None # 
        self.domain_integrators = [] #
        self.boundary_integrators = [] #

    def add_domain_integrator(self, I):
        """
        @brief 增加一个区域积分对象
        """
        self.domain_integrators.append(integ)


    def add_boundary_integrator(self, I):
        """
        @brief 增加一个边界积分对象
        """
        self.boundary_integrators.append(I)

    def mult(self, x):
        """
        """
        return self.M@x

    def add_mult(self, x, y, a=1.0):
        y += a*(self.M@x)


    def assembly(self):
        """
        @brief 调用积分子组装矩阵
        """



import numpy as np

from fealpy.decorator import cartesian

class Hyperbolic2dPDEData:
    def __init__(self, D = (0, 2), T = (0, 4)):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D 
        self._duration = T 

    def domain(self):
        """
        @brief 空间区间
        """
        return self._domain

    def duration(self):
        """
        @brief 时间区间
        """
        return self._duration 
        
    @cartesian
    def solution(self, p, t):
        """
        @brief 真解函数

        @param[in] p numpy.ndarray
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        val = np.zeros_like(p)
        x = p[..., 0]
        y = p[..., 1]
        product = x * y
        flag1 = product <= t
        flag2 = product > t+1
        flag3 = ~flag1 & ~flag2
        
        val[flag1] = 1
        val[flag3] = 1 - product[flag3] + t
        val[flag2] = product[flag2] - t - 1
        
        return val

    def init_solution(self, p):
        """
        @brief 初始解

        @param[in] p numpy.ndarray

        @return 初始解函数值
        """
        val = np.zeros_like(p)
        x = p[..., 0]
        y = p[..., 1]
        val = np.abs(x*y-1)
        
        return val

    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray
        @param[in] t float, 时间点 
        """
        val = np.zeros_like(p)
        x = p[..., 0]
        y = p[..., 1]
        flag1 = np.logical_or(x == 0, y == 0)
        val[flag1] = 1
        return val

    def a(self):
        """
        @brief 返回参数 a 的值
        """
        return 1

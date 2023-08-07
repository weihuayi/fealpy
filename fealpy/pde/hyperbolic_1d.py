import numpy as np

from fealpy.decorator import cartesian

class Hyperbolic1dPDEData:
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

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        val = np.zeros_like(p)
        flag1 = p <= t
        flag2 = p > t+1
        flag3 = ~flag1 & ~flag2
        
        val[flag1] = 1
        val[flag3] = 1 - p[flag3] + t
        val[flag2] = p[flag2] - t - 1
        
        return val

    @cartesian
    def init_solution(self, p):
        """
        @brief 初始解

        @param[in] p numpy.ndarray, 空间点

        @return 初始解函数值
        """
        val = np.zeros_like(p)
        val = np.abs(p-1)
        
        return val

    @cartesian    
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return np.ones(p.shape)

    @cartesian    
    def is_dirichlet_boundary(self, p):
        """
        @brief 判断给定的点是否在 Dirichlet 边界上

        @param[in] p numpy.ndarray, 空间点

        @return bool, 如果 p 在 Dirichlet 边界上则返回 True, 否则返回 False
        """
        return np.isclose(p, self._domain[0])

    def a(self):
        return 1


class Hyperbolic1dSinData:
    def __init__(self, D = (0, 1), T = (0, 0.5)):
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
        return 1 + np.sin(2 * np.pi * (p + 2 * t))

    @cartesian
    def init_solution(self, p):
        """
        @brief 初始解

        @param[in] p numpy.ndarray, 空间点

        @return 初始解函数值
        """
        return 1 + np.sin(2 * np.pi * p)

    @cartesian    
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return 1 + np.sin(4 * np.pi * t)

    @cartesian    
    def is_dirichlet_boundary(self, p):
        """
        @brief 判断给定的点是否在 Dirichlet 边界上

        @param[in] p numpy.ndarray, 空间点

        @return bool, 如果 p 在 Dirichlet 边界上则返回 True, 否则返回 False
        """
        return np.isclose(p, self._domain[1])

    def a(self):
        return -2

import numpy as np

from fealpy.decorator import cartesian
from typing import Tuple

class Hyperbolic1dPDEData:
    def __init__(self, D: Tuple[int, int]=(0, 2), T: Tuple[int, int]=(0, 4)):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D 
        self._duration = T 

    def domain(self) -> Tuple[int, int]:
        """
        @brief 空间区间
        """
        return self._domain

    def duration(self)-> Tuple[int, int]:
        """
        @brief 时间区间
        """
        return self._duration 
        
    @cartesian
    def solution(self, p: np.ndarray, t: np.float64) -> np.ndarray:
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
    def init_solution(self, p: np.ndarray) -> np.ndarray:
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点

        @return 真解函数值
        """
        return np.abs(p-1)
        
    @cartesian
    def source(self, p: np.ndarray, t: np.float64) -> np.float64:
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        return 0.0

    @cartesian    
    def dirichlet(self, p: np.ndarray, t: np.float64) -> np.ndarray:
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return np.ones(p.shape)

    def is_dirichlet_boundary(self, p):
        return np.abs(p - self._domain[0]) < 1e-12

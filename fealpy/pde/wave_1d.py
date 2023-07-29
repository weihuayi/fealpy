import numpy as np

from typing import List, Optional
from ..decorator import cartesian

class StringOscillationPDEData:
    def __init__(self, D = [0, 1], T = [0, 4]):
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
    def init_solution(self, p):
        """
        @brief 初始解

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        val = np.zeros_like(p)
        flag = p < 0.7
        val[flag] = 0.5/7.0*p[flag] 
        val[~flag] = 0.5/3.0*(1 - p[~flag])
        return val 

    @cartesian
    def init_solution_diff_t(self, p):
        """
        @brief 初始解的时间导数函数(初始速度条件)

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解时间导数函数值
        """
        return np.zeros_like(p)

    @cartesian
    def source(self, p, t):
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点

        @return 方程右端函数值
        """
        return np.zeros_like(p)
    
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点

        """
        return np.zeros_like(p) 


class StringOscillationSinCosPDEData:
    def __init__(self, D = [0.0, 1.0], T = [0.0, 2.0]):
        """
        @brief 模型初始化函数

        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D 
        self._duration = T 

    def domain(self) -> List[float]:
        """
        @brief 空间区间
        """
        return self._domain

    def duration(self) -> List[float]:
        """
        @brief 时间区间
        """
        return self._duration 

    @cartesian
    def source(self, p: np.ndarray, t: np.float64) -> float:
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        return 0.0

    def solution(self, x: np.ndarray, t: np.float64) -> np.ndarray:
        """
        @brief 真解函数

        @param[in] x numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        return (np.sin(np.pi*(x - t)) + np.sin(np.pi*(x + t))) / 2 - (np.sin(np.pi*(x - t)) - np.sin(np.pi*(x + t))) / (2 * np.pi)

    def init_solution(self, x: np.ndarray) -> np.ndarray:
        """
        @brief 初始条件函数

        @param[in] x numpy.ndarray, 空间点

        @return 初始条件函数值
        """
        return np.sin(np.pi * x)

    def init_solution_diff_t(self, x: np.ndarray) -> np.ndarray:
        """
        @brief 初始条件时间导数函数(初始速度条件)

        @param[in] x numpy.ndarray, 空间点

        @return 初始条件时间导数函数值
        """
        return np.cos(np.pi * x)

    #def dirichlet(self, x: np.ndarray, t: np.float64) -> np.ndarray:
    #    """
    #    @brief Dirichlet
    #    """
     #   val = np.zeros_like(x)
      #  val[0] = np.sin(np.pi * t) / np.pi
       # val[-1] = -np.sin(np.pi * t) / np.pi
        #return val
 

    def dirichlet(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """
        @brief Dirichlet
        """
        if t is None:
            t = 1.0  # 或者其他适当的默认值
        val = np.zeros_like(x)
        val[0] = np.sin(np.pi * t) / np.pi
        val[-1] = -np.sin(np.pi * t) / np.pi
        return val

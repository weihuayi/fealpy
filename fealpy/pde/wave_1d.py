import numpy as np

from typing import Tuple
from ..decorator import cartesian

class StringOscillationPDEData:
    def __init__(self, D: Tuple[np.float64, np.float64] = [0, 1], 
                T: Tuple[np.float64, np.float64] = [0, 4]):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D 
        self._duration = T 

    def domain(self) -> Tuple[np.float64, np.float64]:
        """
        @brief 空间区间
        """
        return self._domain

    def duration(self) -> Tuple[np.float64, np.float64]:
        """
        @brief 时间区间
        """
        return self._duration 
        
    @cartesian
    def solution(self, p: np.ndarray, t: np.float64) -> np.float64:
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        return 0.0

    @cartesian
    def init_solution(self, p: np.ndarray) -> np.ndarray:
        """
        @brief 初始条件函数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始条件函数值
        """
        val = np.zeros_like(p)
        flag = p < 0.7
        val[flag] = 0.5/7.0*p[flag] 
        val[~flag] = 0.5/3.0*(1 - p[~flag])
        return val 

    @cartesian
    def init_solution_diff_t(self, p: np.ndarray) -> np.ndarray:
        """
        @brief 初始条件时间导数函数(初始速度条件)

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始条件时间导数函数值
        """
        return np.zeros_like(p)

    @cartesian
    def left_solution(self, t: np.float64) -> np.float64:
        """
         @brief 边界条件

         @param[in] t float, 时间点 
        """
        return 0.0

    @cartesian
    def right_solution(self, t: np.float64) -> np.float64:
        """
         @brief 边界条件

         @param[in] t float, 时间点 
        """
        return np.zeros_like(p) 
        
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
    def gradient(self, p: np.ndarray, t: np.float64) -> np.float64:
        """
        @brief 真解导数 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        return 0.0
    
    @cartesian    
    def dirichlet(self, p: np.ndarray, t: np.float64) -> np.float64:
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 边界条件函数值
        """
        return 0.0 

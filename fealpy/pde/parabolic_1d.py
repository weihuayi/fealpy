import numpy as np
from ..decorator import cartesian

class SinExpPDEData:
    def __init__(self, D=[0, 1], T=[0, 1]):
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
        pi = np.pi
        return np.sin(4*pi*p)*np.exp(-10*t) 

    @cartesian
    def init_solution(self, p):
        """
        @brief 初始解

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        pi = np.pi
        return np.sin(4*pi*p) 
        
    @cartesian
    def source(self, p, t):
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        pi = np.pi
        return -10*np.exp(-10*t)*np.sin(4*pi*p) + 16*pi**2*np.exp(-10*t)*np.sin(4*pi*p)
    
    @cartesian
    def gradient(self, p, t):
        """
        @brief 真解导数 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        pi = np.pi
        return 4*pi*np.exp(-10*t)*np.cos(4*pi*p)

    
    @cartesian    
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return self.solution(p, t)


class HeatConductionPDEData:

    def __init__(self, D=[0, 1], T=[0, 1], k=1):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        @param[in] k 热传导系数
        """
        self._domain = D
        self._duration = T
        self._k = k
        self._L = D[1] - D[0] 

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

        @param[in] p float, 空间点
        @param[in] t float, 时间点

        @return 真解函数值
        """
        return np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.sin(np.pi * p / self._L)

    @cartesian
    def init_solution(self, p):
        """
        @brief 初始解函数

        @param[in] x float, 空间点

        @return 初始解函数值
        """
        return np.sin(np.pi * p / self._L)

    @cartesian
    def source(self, p, t):
        """
        @brief 方程右端项

        @param[in] p float, 空间点
        @param[in] t float, 时间点

        @return 方程右端函数值
        """
        return (self._k * (np.pi**2) / self._L**2) * np.exp(-self._k *
                (np.pi**2) * t / self._L**2) * np.sin(np.pi * p / self._L)

    @cartesian
    def gradient(self, p, t):
        """
        @brief 真解空间导数

        @param[in] p float, 空间点
        @param[in] t float, 时间点

        @return 真解空间导函数值
        """
        return (np.pi / self._L) * np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.cos(np.pi * p / self._L)

    @cartesian
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p float, 空间点
        @param[in] t float, 时间点
        """
        return self.solution(p, t)

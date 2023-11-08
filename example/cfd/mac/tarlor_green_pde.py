#!/usr/bin/python3
import numpy as np
class taylor_greenData:
    def __init__(self, D=[0,2*np.pi,0,2*np.pi], T=[0, 5]):
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
    
    def solution_u(self, m, t):
        """
        @brief 真解函数u

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.cos(x)*np.sin(y)*Ft
        return val

    def init_solution_u(self, m):
        """
        @brief 初始解

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        x = m[..., 0]
        y = m[..., 1]
        val = np.cos(x)*np.sin(y)
        return val
    
    def solution_v(self, m, t):
        """
        @brief 真解函数u

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = -np.sin(x)*np.cos(y)*Ft
        return val

    def init_solution_v(self, m):
        """
        @brief 初始解

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        x = m[..., 0]
        y = m[..., 1]
        val = -np.sin(x)*np.cos(y)
        return val
    
    def solution_p(self, m, t):
        """
        @brief 真解函数p

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = -((np.cos(2*x)+np.cos(2*y))*Ft**2)/4
        return val
    
    def init_solution_p(self, m):
        """
        @brief 初始解

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        x = m[..., 0]
        y = m[..., 1]
        val = -(np.cos(2*x)+np.cos(2*y))/4
        return val
    
    def source_F(self,nu,t):
        """
        @brief 体积力
        @param[in] t float, 时间点 
        """
        return np.exp(-2*nu*t)

    def source(self, m, t):
        """
        @brief 方程右端项 

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.zeros(m.shape, dtype=np.float64)
        val[..., 0] = -2*nu*np.cos(x)*np.sin(y)*Ft+np.sin(2*x)*Ft**2
        val[..., 1] = 2*nu*np.sin(x)*np.cos(y)*Ft+np.sin(2*y)*Ft**2
        return val
    
    def gradient_u(self, m, t):
        """
        @brief 真解u导数 

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.zeros(m.shape, dtype=np.float64)
        val[..., 0] = -np.sin(x)*np.sin(y)*Ft
        val[..., 1] = np.cos(x)*np.cos(y)*Ft
        return val
    
    def gradient_u(self, m, t):
        """
        @brief 真解u导数 

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.zeros(m.shape, dtype=np.float64)
        val[..., 0] =-np.cos(x)*np.cos(y)*Ft
        val[..., 1] = np.sin(x)*np.sin(y)*Ft
        return val

    def gradient_p(self, m, t):
        """
        @brief 真解p导数 

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.zeros(m.shape, dtype=np.float64)
        val[..., 0] = (np.sin(2*x)*Ft**2)/2
        val[..., 1] = (np.sin(2*y)*Ft**2)/2
        return val
    
    def dirichlet_u(self, m, t):
        """
        @brief Dirichlet 边界条件

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return self.solution_u(m, t)
    
    def dirichlet_v(self, m, t):
        """
        @brief Dirichlet 边界条件

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return self.solution_v(m, t)



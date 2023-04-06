import numpy as np
import matplotlib.pyplot as plt

from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh1d

class ParabolicPDEModel:

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
        return np.sin(4*pi*p)*np.exp(10*t) 

    @cartesian
    def init_solution(self, p):
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        return np.sin(4*pi*p) 
        
    @cartesian
    def source(self, p):
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        return 5*np.exp(5*t)*np.sin(4*np.pi*x) + 16*np.pi**2*np.exp(5*t)*np.sin(4*np.pi*x)
    
    @cartesian
    def gradient(self, p, t):
        """
        @brief 真解导数 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        return 4*np.pi*np.cos(4*np.pi*p)
    
    @cartesian    
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return self.solution(p, t)


pde = ParabolicPDEModel()

domain = pde.domain()

nx = 10
hx = (domain[1] - domain[0])/nx

mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

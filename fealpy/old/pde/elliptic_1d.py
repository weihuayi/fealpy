import numpy as np

from ..decorator import cartesian
from typing import List 

class SinPDEData:
    """
    -u''(x) = 16 pi^2 sin(4 pi x), 
       u(0) = 0, u(1) = 0.
    exact solution：
       u(x) = sin(4 pi x).
    """
    def domain(self) -> List[int]:
        """
        @brief:    Get the domain of the PDE model

        @return:   A list representing the domain of the PDE model
        """
        return [0, 1]

    @cartesian    
    def solution(self, p: np.ndarray) -> np.ndarray:
        """
        @brief:    Calculate the exact solution of the PDE model

        @param p:  An array of the independent variable x
        @return:   The exact solution of the PDE model at the given points
        """
        return np.sin(4*np.pi*p)
    
    @cartesian    
    def source(self, p: np.ndarray) -> np.ndarray:
        """
        @brief:    Calculate the source term of the PDE model

        @param p:  An array of the independent variable x
        @return:   The source term of the PDE model at the given points
        """
        return 16*np.pi**2*np.sin(4*np.pi*p)
    
    @cartesian    
    def gradient(self, p: np.ndarray) -> np.ndarray:
        """
        @brief:    Calculate the gradient of the exact solution of the PDE model

        @param p:  An array of the independent variable x

        @return:   The gradient of the exact solution of the PDE model at the given points
        """
        return 4*np.pi*np.cos(4*np.pi*p)

    @cartesian    
    def dirichlet(self, p: np.ndarray) -> np.ndarray:
        """
        @brief: Dirichlet BC
        """
        return self.solution(p)


class ExpPDEData:
    def domain(self):
        """
        @brief 得到 PDE 模型的区域

        @return: 表示 PDE 模型的区域的列表
        """
        return [-1, 1]

    @cartesian
    def solution(self, p):
        """
        @brief 计算 PDE 模型的精确解
        
        @param p: 自标量 x 的数组

        @return: PDE 模型在给定点的精确解
        """
        return (np.e**(-p**2))*(1-p**2)

    @cartesian
    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项 

        @param p: 自标量 x 的数组

        @return: PDE 模型在给定点处的源项
        """
        return (np.e**(-p**2))*(4*p**4-16*p**2+6)

    @cartesian    
    def gradient(self, p):
        """
        @brief: 计算 PDE 模型的真解的梯度

        @param p: 自标量 x 的数组

        @return: PDE 模型在给定点处真解的梯度
        """
        return (np.e**(-p**2))*(2*p**3-4*p)

    @cartesian    
    def dirichlet(self, p):
        """
        @brief: 模型的 Dirichlet 边界条件
        """
        return self.solution(p)

class CDRPDEData:
    """
    5u''(x) + u'(x) + 0.001u(x) = q(x)
    BC: u(0) = 100, u'(-1) = 0
    源项 q(x) = 10, x = 0,
              = o, x \neq 0
    """
    def __init__(self):
        self.L = 1000 # 河流的长度
        self.Q0 = 10 # 污染物产生速度
        self.A = 1 # 河流横截面积
        self.C0 = 100 # 河流起点污染物浓度

    def domain(self) -> List[int]:
        return [0, self.L]

    @cartesian
    def source(self, p: np.ndarray) -> np.ndarray:
        Q = np.zeros_like(p)
        Q[1] = self.Q0/self.A
        return Q

    @cartesian    
    def dirichlet(self, x: np.ndarray) -> np.ndarray:
        """
        Dirichlet bc
        """
        return np.where(x==0, self.C0, 0)

    @cartesian
    def is_dirichlet_boundary(self, x: np.ndarray) -> np.ndarray:
        """
        判断给定点是否在 Dirichlet 边界上
        """
        return x == 0

    @cartesian
    def neumann(self, x: np.ndarray) -> np.ndarray:
        """ 
        Neumann bc
        """
        return np.zeros_like(x)

    @cartesian
    def is_neumann_boundary(self, x: np.ndarray) -> np.ndarray:
        """
        判断给定点是否在 Neumann 边界上
        """
        return x == self.L

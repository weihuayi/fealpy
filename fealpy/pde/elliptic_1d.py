import numpy as np
from ..decorator import cartesian

class SinPDEData:
    """
    -u''(x) = 16\pi^2\sin(4\pi x), 
       u(0) = 0,\quad u(1) = 0.
    真解：u(x) = \sin(4\pi x).
    """
    def domain(self):
        """
        @brief:    Get the domain of the PDE model

        @return:   A list representing the domain of the PDE model
        """
        return [0, 1]

    @cartesian    
    def solution(self, p):
        """
        @brief: Calculate the exact solution of the PDE model

        @param p: An array of the independent variable x
        @return: The exact solution of the PDE model at the given points
        """
        return np.sin(4*np.pi*p)
    
    @cartesian    
    def source(self, p):
        """
        @brief: Calculate the source term of the PDE model

        @param p: An array of the independent variable x
        @return: The source term of the PDE model at the given points
        """
        return 16*np.pi**2*np.sin(4*np.pi*p)
    
    @cartesian    
    def gradient(self, p):
        """
        @brief: Calculate the gradient of the exact solution of the PDE model

        @param p: An array of the independent variable x

        @return: The gradient of the exact solution of the PDE model at the given points
        """
        return 4*np.pi*np.cos(4*np.pi*p)

    @cartesian    
    def dirichlet(self, p):
        """
        @brief: 模型的 Dirichlet 边界条件
        """
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        return x == 0.0

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE)
        n: (NE)

        grad*n : (NQ, NE)
        """
        grad = self.gradient(p)
        val = np.sum(grad*n, axis=-1)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        return x == 1.0


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

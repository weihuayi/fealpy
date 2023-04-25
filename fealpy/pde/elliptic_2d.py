import numpy as np
from ..decorator import cartesian


class SinSinPDEData:
    def domain(self):
        """
        @brief 得到 PDE 模型的区域
        @return: 表示 PDE 模型的区域的列表
        """
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """
        @brief 计算 PDE 模型的精确解
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点的精确解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.sin(pi*y)
        return val 
    
    @cartesian
    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项 
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处的源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)
        return val
    
    @cartesian
    def gradient(self, p):
        """
        @brief: 计算 PDE 模型的真解的梯度
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处真解的梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        return val

    @cartesian    
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return np.abs(x) < 1e-12 | np.abs(y) < 1e-12

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return np.abs(x-1.0) < 1e-12 | np.abs(y-1.0) < 1e-12

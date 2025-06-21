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

class CosCosPDEData:
    def domain(self):
        return np.array([0, 1, 0, 1])
    
    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.sin(pi*y)-np.cos(pi*x)*np.cos(pi*y)+1
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi**2*(np.sin(pi*x)*np.sin(pi*y)-np.cos(pi*x)*np.cos(pi*y))
        return val

    @cartesian
    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)+pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)+pi*np.cos(pi*x)*np.sin(pi*y)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

class PoissonPDEData:
    """
    真解为 u(x, y) = (\cos\pi x \cos\pi y) 的 Poisson 方程
    """
    def domain(self):
        return np.array([0, 1, 0, 1])
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi**2*np.cos(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)


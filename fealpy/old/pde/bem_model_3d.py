import numpy as np

from ..decorator import cartesian


class PoissonModelConstantDirichletBC3d:
    """
    三维常单元纯 Dirichlet 边界 Poisson 方程模型，其真解为 $u(x, y)=\cos(\pi*x)*\cos(\pi*y)*\cos(\pi*z)$
    """
    def domain(self):
        """
        @brief 得到 PDE 模型的区域
        @return: 表示 PDE 模型的区域的列表
        """
        return np.array([0, 1, 0, 1, 0, 1])

    def solution(self, p):
        """
        @brief 计算 PDE 模型的精确解
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点的精确解
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        val = np.cos(pi * x) * np.cos(pi * y) * np.cos(pi * z)
        return val

    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处的源项
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        val = 3 * pi * pi * np.cos(pi * x) * np.cos(pi * y) * np.cos(pi * z)
        return val

    def gradient(self, p):
        """
        @brief: 计算 PDE 模型的真解的梯度
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处真解的梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = - pi * np.sin(pi * x) * np.cos(pi * y) * np.cos(pi * z)
        val[..., 1] = - pi * np.cos(pi * x) * np.sin(pi * y) * np.cos(pi * z)
        val[..., 2] = - pi * np.cos(pi * x) * np.cos(pi * y) * np.sin(pi * z)
        return val

    def dirichlet(self, p):
        return self.solution(p)

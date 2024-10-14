import numpy as np

from ..decorator import cartesian


class LaplaceBemModelMixedBC2d:
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
        val = 100 * (1 + x)
        return val

    @cartesian
    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处的源项
        """
        val = np.zeros_like(p)
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
        val = np.zeros((p.shape[0], 2), dtype=np.float64)
        val[..., 0] = 100
        val[..., 1] = 0
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return (np.abs(x - 1.0) < 1e-12) | (np.abs(x - 0.0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        grad = self.gradient(p)
        val = np.sum(grad * n, axis=-1)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return (np.abs(y - 1.0) < 1e-12) | (np.abs(y - 0.0) < 1e-12)


class LaplaceBemModelDirichletBC2d:
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
        val = x - y
        return val

    @cartesian
    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处的源项
        """
        val = np.zeros_like(p)
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
        val = np.zeros((p.shape[0], 2), dtype=np.float64)
        val[..., 0] = 1
        val[..., 1] = -1
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return (np.abs(x - 1.0) < 1e-12) | (np.abs(x - 0.0) < 1e-12) | (np.abs(y - 1.0) < 1e-12) | (
                    np.abs(y - 0.0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        grad = self.gradient(p)
        val = np.sum(grad * n, axis=-1)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return np.zeros(p.shape[0], dtype=bool)


class LaplaceBemModelNeumannBC2d:
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
        val = x - y
        return val

    @cartesian
    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处的源项
        """
        val = np.zeros_like(p)
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
        val = np.zeros((p.shape[0], 2), dtype=np.float64)
        val[..., 0] = 1
        val[..., 1] = -1
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return np.zeros(p.shape[0], dtype=bool)

    @cartesian
    def neumann(self, p, n):
        grad = self.gradient(p)
        val = np.sum(grad * n, axis=-1)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return (np.abs(x - 1.0) < 1e-12) | (np.abs(x - 0.0) < 1e-12) | (np.abs(y - 1.0) < 1e-12) | (
                    np.abs(y - 0.0) < 1e-12)


class PoissonModelConstantDirichletBC2d:
    """
    二维常单元纯 Dirichlet 边界 Poisson 方程模型，其真解为 $u(x, y)=\sin(\pi*x)*\sin(\pi*y)$
    """

    def domain(self):
        """
        @brief 得到 PDE 模型的区域
        @return: 表示 PDE 模型的区域的列表
        """
        return np.array([0, 1, 0, 1])

    def solution(self, p):
        """
        @brief 计算 PDE 模型的精确解
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点的精确解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi * x) * np.sin(pi * y)
        return val

    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处的源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2 * pi * pi * np.sin(pi * x) * np.sin(pi * y)
        return val

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
        val[..., 0] = - pi * np.cos(pi * x) * np.sin(pi * y)
        val[..., 1] = - pi * np.sin(pi * x) * np.cos(pi * y)
        return val

    def dirichlet(self, p):
        return self.solution(p)

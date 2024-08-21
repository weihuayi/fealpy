
from ..backend import backend_manager as bm
from ..decorator import cartesian, barycentric

class CosCosData:
    """
        -\\Delta u = f
        u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self, kappa=1.0):
        self.kappa = kappa # Robin 条件中的系数

    def domain(self):
        """
        @brief 模型定义域
        """
        return [0, 1, 0, 1]

    @cartesian
    def solution(self, p):
        """  
        @brief 模型真解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.cos(pi*x)*bm.cos(pi*y)
        return val # val.shape == x.shape


    @cartesian
    def source(self, p):
        """ 
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = 2*pi*pi*bm.cos(pi*x)*bm.cos(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        """  
        @brief 真解梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        val = bm.stack((
            -pi*bm.sin(pi*x)*bm.cos(pi*y),
            -pi*bm.cos(pi*x)*bm.sin(pi*y)), axis=-1)
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        """
        @brief 真解通量
        """
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件 
        """
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        """
        y = p[..., 1]
        return (bm.abs(y - 1.0) < 1e-12) | (bm.abs( y -  0.0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        """ 
        @brief Neumann 边界条件
        """
        grad = self.gradient(p) # (NF, NQ, 2)
        if n.ndim == 2:
            n = bm.expand_dims(n, axis=1)
        val = bm.einsum('fqd, fqd -> fq', grad, n) # (NF, NQ)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        """
        @brief Neumann 边界的判断函数
        """
        x = p[..., 0]
        return bm.abs(x - 1.0) < 1e-12

    @cartesian
    def robin(self, p, n):
        """
        @brief Robin 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = bm.sum(grad*n, axis=-1)
        val += self.kappa*self.solution(p) 
        return val

    @cartesian
    def is_robin_boundary(self, p):
        """
        @brief Robin 边界条件判断函数
        """
        x = p[..., 0]
        return bm.abs(x - 0.0) < 1e-12

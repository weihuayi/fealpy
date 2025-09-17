#!/usr/bin/python3

from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike
from fealpy.decorator import cartesian

class CosCosData2D:
    """
    Darcy model:
    u + \\nabla p = f
    \\nabla \cdot u = 0
    u \cdot n = 0
    真解：
        u = (sin(πx)cos(πy), -cos(πx)sin(πy))
        p = cos(πx)cos(πy)

    域：Ω = [0, 1]^2
    """
    
    def geo_dimension(self) -> int:
        """
        Return the geometric dimension of the domain.
        """
        return 2
    
    def domain(self) -> Sequence[float]:
        return [0.0, 1.0 ,0.0 ,1.0]

    @cartesian
    def mu_coef(self, p:TensorLike) -> TensorLike:
        """
        Return the computational domain [xmin, xmax, ymin, ymax].
        """
        return 2
    
    def beta(self) -> TensorLike:
        return bm.tensor([0.0])
    
    @cartesian
    def velocity(self, p:TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        u1 = bm.sin(pi * x) * bm.cos(pi * y)
        u2 = -bm.cos(pi * x) * bm.sin(pi * y)
        return bm.stack([u1, u2], axis=-1)  # shape (..., 2)
    
    @cartesian
    def pressure(self, p:TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        return bm.cos(pi * x) * bm.cos(pi * y)
    
    @cartesian
    def grad_pressure(self, p:TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        grad_px = - pi*bm.sin(pi*x)*bm.cos(pi*y)
        grad_py = - pi*bm.cos(pi*x)*bm.sin(pi*y)
        return bm.stack([grad_px, grad_py], axis=-1)
    
    @cartesian
    def f(self, p:TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        mu = self.mu_coef(p)
        u = self.velocity(p)
        grad_p = self.grad_pressure(p)
        
        return mu*u + grad_p
    
    @ cartesian
    def g(self, p:TensorLike) -> TensorLike:
        return bm.zeros_like(p[..., 0])
    
    @cartesian
    def is_neumann_boundary(self, p:TensorLike) -> TensorLike:
        """
        全边界上，u ⋅ n = 0（Neumann-like 边界）
        """
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (
            (bm.abs(x) < atol) | (bm.abs(x-1) < atol) |
            (bm.abs(y) < atol) | (bm.abs(y-1) < atol))
    
    @cartesian
    def neumann(self, p:TensorLike, n:TensorLike) -> TensorLike:
        """
        u ⋅ n = 0
        实际模拟中，这个约束需要以弱形式嵌入，比如混合有限元。
        此函数保留接口。
        """
        u = self.velocity(p)
        if n.ndim == 2:
            n = bm.expand_dims(n, axis=1)
        return bm.einsum("fqd,fqd->fq", u, n)
        
        
    
    
    
    
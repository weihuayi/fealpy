

from typing import Optional, Sequence
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d


class Exp0011(BoxMesher2d):

    def __init__(self):
        # physical / numerical parameters
        self.box = [-2, 2, -2, 2.0]
        super().__init__(self.box)
        self.mu = 1
        self.beta = 0.0
        self.rho = 1.0
    
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
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:

        x, y = p[..., 0], p[..., 1]
        xmin, xmax, ymin, ymax = self.box
        left   = bm.abs(x - xmin) < self.tol
        right  = bm.abs(x - xmax) < self.tol
        bottom = bm.abs(y - ymin) < self.tol
        top    = bm.abs(y - ymax) < self.tol
        return left | right | bottom | top
    
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
        
        
    
    
    
    
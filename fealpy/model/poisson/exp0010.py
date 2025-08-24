import sympy as sp

from sympy.utilities.lambdify import lambdify
from typing import Sequence

from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import BoxMesher2d

class Exp0010(BoxMesher2d):
    """
    2D Poisson problem:
    
        -Δu(x,y) = f(x,y),   (x,y) ∈ (0,1)^2 
        u(x, y) = g(x, y),    on ∂Ω
    
    with the exact solution:
    
        u(x,y) = -[1.5cos(πx + 2π/5) + 2cos(2πx - π/5)] 
                 *[1.5cos(πy + 2π/5) + 2cos(2πy - π/5)]

    Homogeneous Dirichlet boundary conditions are applied on all edges.

    Reference:
        https://doi.org/10.48550/arXiv.2207.13380
    """
    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0] 
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x, y = p[..., 0], p[..., 1]
        val = -((3/2) * bm.cos(bm.pi * x + 2*bm.pi/5) + 2 * bm.cos(2*bm.pi * x - bm.pi/5)) \
            * ((3/2) * bm.cos(bm.pi * y + 2*bm.pi/5) + 2 * bm.cos(2*bm.pi * y - bm.pi/5))
        return val

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos
        t1 = - 1.5 * cos(pi * x + 2*pi/5) + 2 * cos(2*pi * x - pi/5)
        t2 = 1.5 * cos(pi * y + 2*pi/5) + 2 * cos(2*pi * y - pi/5)
        dx = pi * (1.5 * sin(pi * x + 2*pi/5) + 4 * sin(2*pi * x - pi/5)) * t2
        dy = -pi * (1.5 * sin(pi * y + 2*pi/5) + 4 * sin(2*pi * y - pi/5)) * t1
        val = bm.stack([dx, dy], axis=-1)
        return val

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x, y = p[..., 0], p[..., 1]      
        pi = bm.pi
        cos = bm.cos
        t1 = 1.5 * cos(pi * x + 2*pi/5) + 2 * cos(2*pi * x - pi/5)
        t2 = 1.5 * cos(pi * y + 2*pi/5) + 2 * cos(2*pi * y - pi/5)
        dxx = -pi**2 * (1.5 * cos(pi * x + 2*pi/5) + 8 * cos(2*pi * x - pi/5)) * t2
        dyy = -pi**2 * (1.5 * cos(pi * y + 2*pi/5) + 8 * cos(2*pi * y - pi/5)) * t1
        val = dxx + dyy
        return val

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12  # 绝对误差容限
        on_boundary = (
            (bm.abs(x - 1.) < atol) | (bm.abs(x) < atol) |
            (bm.abs(y - 1.) < atol) | (bm.abs(y) < atol)
        )
        return on_boundary 
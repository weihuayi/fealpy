import sympy as sp

from sympy.utilities.lambdify import lambdify
from typing import Sequence

from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import IntervalMesher

class Exp0006(IntervalMesher):
    """
    1D Helmholtz problem:
    
        -Δu(x) - k^2*u(x) = f(x),   x ∈ [0,8] 
          u(x) = g(x),    on ∂Ω
    
    with the exact solution:
    
        u(x) = sin(3πx + 3π/20)*cos(2πx + π/10) + 2

    Homogeneous Dirichlet boundary conditions are applied on all edges.
    """
    def __init__(self, options: dict = {}):
        self.box = [0.0, 8.0] 
        super().__init__(interval=self.box)
        self.k = bm.tensor(options.get('k', 1.0))

    def get_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        # u_func = lambdify((self.x), self.u, bm.backend_name)
        x = p
        val = bm.sin(3*bm.pi*x + 3*bm.pi/20) * bm.cos(2*bm.pi*x + bm.pi/10) + 2
        return val

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        # grad_func = lambdify((self.x), dx, bm.backend_name)
        x = p
        t1 = 3*bm.pi*bm.cos(3*bm.pi*x + 3*bm.pi/20) * bm.cos(2*bm.pi*x + bm.pi/10) # 3*pi*cos(2*pi*x + pi/10)*cos(3*pi*x + 3*pi/20)
        t2 = -2*bm.pi*bm.sin(3*bm.pi*x + 3*bm.pi/20) * bm.sin(2*bm.pi*x + bm.pi/10) # -2*pi*sin(2*pi*x + pi/10)*sin(3*pi*x + 3*pi/20)
        val = t1 + t2
        return val

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        # -pi**2*(12*sin(pi*(2*x + 1/10))*cos(pi*(3*x + 3/20)) + 13*sin(pi*(3*x + 3/20))*cos(pi*(2*x + 1/10)))
        x = p
        t1 = bm.sin(3*bm.pi*x + 3*bm.pi/20)
        t2 = bm.cos(3*bm.pi*x + 3*bm.pi/20)
        t3 = bm.sin(2*bm.pi*x + bm.pi/10)
        t4 = bm.cos(2*bm.pi*x + bm.pi/10)
        laplace = -bm.pi**2 * (12 * t3 * t2 + 13 * t1 * t4)
        val = -laplace - self.k**2 * self.solution(p)
        return val

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        atol = 1e-12  # 绝对误差容限
        on_boundary = (
            (bm.abs(x - 8.) < atol) | (bm.abs(x) < atol)
        )
        return on_boundary 
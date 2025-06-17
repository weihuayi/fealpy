from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class SinCosData2D:
    """
    2D wave equation problem:

        ∂²u/∂t² - Δu(x, y) = 0,      (x, y) ∈ (0, 1) x (0, 1), t on (0, 1.4)
        u(x, y, 0) = sin(πx)·sin(πy), initial solution 
        ∂u/∂t(x, y, 0) = 0,           initial velocity
        u = 0,                        on all four edges

    Exact solution:

        u(x, y, t) = cos(√2·π·t)·sin(πx)·sin(πy)

    This problem imposes homogeneous Dirichlet boundary conditions.
    It is suitable for verifying 2D hyperbolic solvers.
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 1.0, 0.0, 1.0]  
    
    def speed(self) -> float:
        """Return propagation speed a."""
        a = 1.0
        return a

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 1.4]  

    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        """Compute initial condition u(x, y, 0) = sin(πx)·sin(πy)."""
        x, y = p[..., 0], p[..., 1]
        return bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

    @cartesian
    def init_solution_t(self, p: TensorLike) -> TensorLike:
        """Compute initial condition ∂u/∂t(x, y, 0) = 0. """
        return bm.zeros_like(p[..., 0])

    @cartesian
    def solution(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact solution at time t. """
        x, y = p[..., 0], p[..., 1]
        return bm.cos(bm.sqrt(2.0) * bm.pi * t) * bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

    @cartesian
    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        """Compute spatial gradient of solution at time t."""
        x, y = p[..., 0], p[..., 1]
        factor = bm.cos(bm.sqrt(2.0) * bm.pi * t)
        dx = bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y) * factor
        dy = bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y) * factor
        return bm.stack([dx, dy], axis=-1)

    @cartesian
    def source(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact source at time t. """
        return bm.zeros_like(p[..., 0])

    @cartesian
    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        """Dirichlet boundary condition. """
        return bm.zeros_like(p[..., 0])

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 1.0) < 1e-12) | \
               (bm.abs(y - 0.0) < 1e-12) | (bm.abs(y - 1.0) < 1e-12)


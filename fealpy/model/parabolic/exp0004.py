from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm
from ...mesher import BoxMesher3d

class Exp0004(BoxMesher3d):
    """
    2D parabolic problem with space-time FEM:

        ∂u/∂t - (∂²u/∂x² + ∂²u/∂y²) = f   (x, y) ∈ (0, 1) x (0, 1), t > 0

        u(x, 0, t) = u(x, 1, t) = 0,         x ∈ (0, 1), t > 0
        u(0, y, t) = u(1, y, t) = 0,         y ∈ (0, 1), t > 0
        u(x, y, 0) = sin(πx)·sin(πy),     (x, y) ∈ (0, 1) x (0, 1)
    
    Exact solution:
        u(x, y, t) = sin(πx)·sin(πy)·exp(-t)
        f(x, y, t) = -exp(-t)·sin(πx)·sin(πy) + 2π²·exp(-t)·sin(πx)·sin(πy)
    This example imposes homogeneous Dirichlet boundary conditions on all four edges.
    It is useful for verifying time-dependent solvers with space-time FEM in 2D.
    """
    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.duration = [0.0, 1.0]
        super().__init__(box=self.box + self.duration)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box 

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 1.0]
    
    @cartesian
    def diffusion_coef(self, p: TensorLike = None , epsilon: float = 0.0) -> float:
        """Return the diffusion coefficient (constant in this case)."""
        return 1.0

    @cartesian
    def convection_coef(self, p: TensorLike = None) -> TensorLike:
        """Return the convection coefficient (zero in this case)."""
        gd = self.geo_dimension()
        b = bm.zeros((gd-1), dtype=bm.float64)
        return b

    @cartesian
    def reaction_coef(self, p: TensorLike = None) -> float:
        """Return the reaction coefficient (zero in this case)."""
        return 0.0

    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        return bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution at time t. """
        x = p[..., 0]
        y = p[..., 1]
        t = p[..., 2]
        return bm.sin(bm.pi * x) * bm.sin(bm.pi * y) * bm.exp(-t)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute spatial gradient of solution at time t."""
        x = p[..., 0]
        y = p[..., 1]
        t = p[..., 2]
        factor = bm.exp(-t)
        gx = bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y) * factor
        gy = bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y) * factor
        gt = -bm.sin(bm.pi * x) * bm.sin(bm.pi * y) * factor
        return bm.stack([gx, gy, gt], axis=-1)  # shape (..., 3)

    @cartesian
    def time_gradient(self, p: TensorLike) -> TensorLike:
        """Compute the time derivative of the solution."""
        x = p[..., 0]
        y = p[..., 1]
        t = p[..., 2]
        return - bm.sin(bm.pi * x) * bm.sin(bm.pi * y) * bm.exp(-t)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source at time t. """
        x = p[..., 0]
        y = p[..., 1]
        t = p[..., 2]
        return (2 * bm.pi ** 2 - 1) * bm.exp(-t) * bm.sin(bm.pi * x) * bm.sin(bm.pi * y)
    
    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)
    
    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        t = p[..., 2]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 1.0) < 1e-12) | \
               (bm.abs(y - 0.0) < 1e-12) | (bm.abs(y - 1.0) < 1e-12) | (bm.abs(t - 0.0) > 1e-12)

    @cartesian
    def is_init_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on initial boundary."""
        t = p[..., 2]
        return bm.abs(t - 0.0) < 1e-12

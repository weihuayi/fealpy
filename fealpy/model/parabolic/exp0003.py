from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm
from ...mesher import BoxMesher2d

class Exp0003(BoxMesher2d):
    """
    1D parabolic problem with space-time FEM:

        ∂u/∂t - ∂²u/∂x² = f,     x ∈ (0, 1), t > 0
        u(0, t) = u(1, t) = 0,   t > 0
        u(x, 0) = sin(πx),      x ∈ (0, 1)

    Exact solution:

        u(x, t) = cos(πt)·sin(πx)
        f(x, t) = (π²·cos(πt) - π·sin(πt))·sin(πx)

    This example imposes homogeneous Dirichlet boundary conditions at both ends.
    It is useful for testing time-dependent solvers with space-time FEM.
    """
    def __init__(self):
        self.interval = [0.0, 1.0]
        self.duration = [0.0, 1.0]
        super().__init__(box=self.interval + self.duration)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax]."""
        return self.interval

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 1.0]

    @cartesian
    def diffusion_coef(self, p: TensorLike = None) -> float:
        """Return the diffusion coefficient (constant in this case)."""
        return 1.0

    @cartesian
    def convection_coef(self, p: TensorLike = None) -> TensorLike:
        """Return the convection coefficient (zero in this case)."""
        return 0.0
    
    @cartesian
    def reaction_coef(self, p: TensorLike = None) -> float:
        """Return the reaction coefficient (zero in this case)."""
        return 0.0

    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        """ Initial condition at t = 0: u(x, 0) = sin(πx)"""
        x = p[..., 0]
        return bm.sin(bm.pi * x)

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution at time t. """
        x = p[..., 0]
        t = p[..., 1]
        return bm.sin(bm.pi * x) * bm.cos(bm.pi * t)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """
        Compute the gradient of the solution.
        Note: If the PDE model is one-dimensional, the tensor returned by 
        the gradient computation should match the shape of the input tensor p.
        """
        x = p[..., 0]
        t = p[..., 1]
        g_x = bm.pi * bm.cos(bm.pi * x) * bm.cos(bm.pi * t)
        g_t = -bm.pi * bm.sin(bm.pi * x) * bm.sin(bm.pi * t)
        return bm.stack([g_x, g_t], axis=-1)

    @cartesian
    def time_gradient(self, p: TensorLike) -> TensorLike:
        """Compute the time derivative of the solution."""
        x = p[..., 0]
        t = p[..., 1]
        return -bm.pi * bm.sin(bm.pi * x) * bm.sin(bm.pi * t)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source at time t. """
        x = p[..., 0]
        t = p[..., 1]
        return (bm.pi ** 2 * bm.cos(bm.pi * t) - bm.pi * bm.sin(bm.pi * t)) * bm.sin(bm.pi * x)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        t = p[..., 1]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 1.0) < 1e-12) | (bm.abs(t - 0.0) > 1e-12)

    @cartesian
    def is_init_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is at initial time t = 0."""
        t = p[..., 1]
        return bm.abs(t - 0.0) < 1e-12

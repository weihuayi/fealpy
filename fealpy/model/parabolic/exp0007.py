from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm
from ...mesher import BoxMesher2d

class Exp0007(BoxMesher2d):
    """
    1D parabolic problem with space-time FEM:

        ∂u/∂t - epsilon*(∂²u/∂x²) + b · ∇u = f   (x) ∈ (0, 1), t > 0

        u(0, t) = u(1, t) =  0,         x ∈ (0, 1), t > 0
        u(x, 0) = sin(pi x)*sin(2pi x),     (x) ∈ (0, 1)
    
    This example imposes homogeneous Dirichlet boundary conditions on all four edges.
    It is useful for verifying time-dependent solvers with space-time FEM in 2D.
    """
    def __init__(self):
        self.box = [0.0, 1.0]
        self.duration = [0.0, 1.0]
        super().__init__(box=self.box + self.duration)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box 

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 1.0]
    
    @cartesian
    def diffusion_coef(self, p: TensorLike = None) -> float:
        """Return the diffusion coefficient (constant in this case)."""
        return 1e-1

    @cartesian
    def convection_coef(self, p: TensorLike = None) -> TensorLike:
        """Return the convection coefficient (zero in this case)."""
        return 1

    @cartesian
    def reaction_coef(self, p: TensorLike = None) -> float:
        """Return the reaction coefficient (zero in this case)."""
        return 0.0

    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        """
        Return the initial solution at time t=0.
        """
        return bm.sin(bm.pi * p[..., 0]) * bm.sin(2 * bm.pi * p[..., 0])

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution at time t. """
        b = 1
        return bm.sin(bm.pi * p[..., 0]) * bm.sin(2 * bm.pi * p[..., 0] - b * p[..., 1])

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute spatial gradient of solution at time t."""
        b= 1
        term0 = bm.pi * bm.cos(bm.pi * p[..., 0]) * bm.sin(2 * bm.pi * p[..., 0] - b * p[..., 1])
        term1 = 2 * bm.pi * bm.sin(bm.pi * p[..., 0]) * bm.cos(2 * bm.pi * p[..., 0] - b * p[..., 1])
        termx = term0 + term1
        termt = -b * bm.sin(bm.pi * p[..., 0]) * bm.cos(2 * bm.pi * p[..., 0] - b * p[..., 1])
        return bm.stack((termx, termt), axis=-1)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source at time t. """
        b = 1
        e = 1e-1
        x = p[..., 0]
        t = p[..., 1]
        theta = 2*bm.pi*x - b*t
        term0 = bm.sin(theta)*(b*bm.pi*bm.cos(bm.pi*x) + e*5*bm.pi**2*bm.sin(bm.pi*x))
        term1 = bm.cos(theta)*(b*(2*bm.pi-1)*bm.sin(bm.pi*x) - e*4*bm.pi**2*bm.cos(bm.pi*x))
        return term0 + term1

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        t = p[..., 1]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 1.0) < 1e-12) & (bm.abs(t - 0.0) > 1e-12)

    @cartesian
    def is_init_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on initial boundary."""
        t = p[..., 1]
        return bm.abs(t - 0.0) < 1e-12
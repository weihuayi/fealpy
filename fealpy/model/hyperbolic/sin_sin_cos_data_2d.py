from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class SinSinCosData2D:
    """
    2D hyperbolic problem with sinusoidal solution:

        ∂u/∂t + a·(∂u/∂x + ∂u/∂y) = f,     x ∈ (0, 2), y ∈ (0, 2), t > 0
        u(x, y, 0) = sin(πx)sin(πy),        initial condition
        u(0, y, t) = u(x, 0, t) = 0,        Dirichlet boundary condition

    Exact solution:
        u(x, y, t) = sin(πx)sin(πy)cos(πt)
    This represents a standing wave pattern in a 2D square domain.
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [0.0, 2.0, 0.0, 2.0]  

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 1.0]  
    
    def convection_coef(self) -> TensorLike:
        """ Get the convection coefficient (wave speed)."""
        return bm.tensor([1.0, 1.0])  

    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        """Compute initial condition ."""
        x, y = p[..., 0], p[..., 1]
        return bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

    @cartesian
    def solution(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact solution at time t. """
        x, y = p[..., 0], p[..., 1]
        return bm.sin(bm.pi * x) * bm.sin(bm.pi * y) * bm.cos(bm.pi * t)

    @cartesian
    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        """Compute spatial gradient of solution at time t."""
        x, y = p[..., 0], p[..., 1]
        cos_πt = bm.cos(bm.pi * t)
        du_dx = bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y) * cos_πt
        du_dy = bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y) * cos_πt
        return bm.stack([du_dx, du_dy], axis=-1)  # Returns a vector field

    @cartesian
    def source(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact solution at time t. """
        x, y = p[..., 0], p[..., 1]
        sin_πt = bm.sin(bm.pi * t)
        term1 = -bm.pi * bm.sin(bm.pi * x) * bm.sin(bm.pi * y) * sin_πt  # ∂u/∂t
        term2 = bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y) * bm.cos(bm.pi * t)  # ∂u/∂x
        term3 = bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y) * bm.cos(bm.pi * t)  # ∂u/∂y
        return term1 + self.convection_coef()[1] * (term2 + term3)

    @cartesian
    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        """Dirichlet boundary condition."""
        return self.solution(p=p, t=t) 

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 2.0) < 1e-12) | \
               (bm.abs(y - 0.0) < 1e-12) | (bm.abs(y - 2.0) < 1e-12)

 
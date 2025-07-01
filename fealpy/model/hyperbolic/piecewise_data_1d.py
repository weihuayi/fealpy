from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class PiecewiseData1D:
    """
    1D hyperbolic problem with piecewise linear solution:

        ∂u/∂t + a·∂u/∂x = 0,     x ∈ (0, 2), t ∈ (0, 4)
        u(x, 0) = |x - 1|,        x ∈ (0, 2)
        u(0, t) = 1,              t ∈ (0, 4)
        a = 1.0
    Exact solution is a piecewise linear function with three regions:
        u = 1 for x ≤ t
        u = 1 - x + t for t < x ≤ t+1
        u = x - t - 1 for x > t+1
    This represents a wave propagating with speed a = 1.0.
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 1

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax]."""
        return [0.0, 2.0]

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 4.0]

    def convection_coef(self) -> TensorLike:
        """ Get the convection coefficient (wave speed)."""
        return bm.tensor([1.0])
    
    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        """Compute initial condition u(x,0) = |x - 1|."""
        x = p[..., 0]
        return bm.abs(x - 1)

    @cartesian
    def solution(self, p: TensorLike, t: float) -> TensorLike:
        """Compute exact solution at time t. """
        x = p[..., 0]
        val = bm.zeros_like(x)
        flag1 = x <= t
        flag2 = x > t + 1
        flag3 = ~flag1 & ~flag2
        val[flag1] = 1
        val[flag3] = 1 - x[flag3] + t
        val[flag2] = x[flag2] - t - 1
        return val

    @cartesian
    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        """
        Compute the gradient of the solution.
        Note: If the PDE model is one-dimensional, the tensor returned by 
        the gradient computation should match the shape of the input tensor p.
        """
        x = p
        grad = bm.zeros_like(x)
        flag1 = x <= t
        flag2 = x > t + 1
        flag3 = ~flag1 & ~flag2
        grad[flag1] = 0
        grad[flag3] = -1
        grad[flag2] = 1
        return grad

    @cartesian
    def source(self, p: TensorLike, t: float) -> TensorLike:
        """Compute source term (always zero for this problem)."""
        x = p[..., 0]
        return bm.zeros_like(x)

    @cartesian
    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        """Dirichlet boundary condition u(0,t)=1"""
        x = p[..., 0]
        return bm.ones_like(x)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 2.0) < 1e-12)


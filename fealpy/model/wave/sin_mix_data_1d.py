from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class SinMixData1D:
    """
    1D wave equation problem:

        ∂²u/∂t² - ∂²u/∂x² = 0,    x ∈ (0, 1), t > 0
        u(0, t) = u(1, t) = 0,    t > 0
        u(x, 0) = sin(4πx),       x ∈ (0, 1)
        ∂u/∂t(x, 0) = sin(8πx),   x ∈ (0, 1)

    Exact solution:

        u(x, t) = cos(4πt)·sin(4πx) + (1 / 8π)·sin(8πt)·sin(8πx)

    This example imposes homogeneous Dirichlet boundary conditions.
    It is useful for testing second-order hyperbolic solvers.
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 1

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax]."""
        return [0.0, 1.0]
    
    def speed(self) -> float:
        """Return propagation speed a."""
        a = 1.0
        return a

    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 1.0]  

    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        """Compute initial condition u(x, 0) = sin(4πx)."""
        x = p[..., 0]
        return bm.sin(4 * bm.pi * x)

    @cartesian
    def init_solution_t(self, p: TensorLike) -> TensorLike:
        """Compute initial condition ∂u/∂t(x, 0) = sin(8πx). """
        x = p[..., 0]
        return bm.sin(8 * bm.pi * x)

    @cartesian
    def solution(self, p: TensorLike, t: float) -> TensorLike:
        """
        Compute the gradient of the solution.
        Note: If the PDE model is one-dimensional, the tensor returned by 
        the gradient computation should match the shape of the input tensor p.
        """
        x = p
        term1 = bm.cos(4 * bm.pi * t) * bm.sin(4 * bm.pi * x)
        term2 = (1 / (8 * bm.pi)) * bm.sin(8 * bm.pi * t) * bm.sin(8 * bm.pi * x)
        return term1 + term2

    @cartesian
    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        """Compute spatial gradient of solution at time t."""
        x = p[..., 0]
        term1 = bm.cos(4 * bm.pi * t) * 4 * bm.pi * bm.cos(4 * bm.pi * x)
        term2 = (1 / (8 * bm.pi)) * bm.sin(8 * bm.pi * t) * 8 * bm.pi * bm.cos(8 * bm.pi * x)
        return term1 + term2  # du/dx

    @cartesian
    def source(self, p: TensorLike, t: float) -> TensorLike:
        "" "Compute exact source at time t. """    
        x = p[..., 0]
        return bm.zeros_like(x)

    @cartesian
    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        """Dirichlet boundary condition. """
        x = p[..., 0]
        return bm.zeros_like(x)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 1.0) < 1e-12)


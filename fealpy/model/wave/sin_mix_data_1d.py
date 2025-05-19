from typing import Sequence
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

    def duration(self) -> Sequence[float]:
        return [0.0, 1.0]  # Time domain: [t0, t1]

    def init_solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return bm.sin(4 * bm.pi * x)

    def init_solution_t(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return bm.sin(8 * bm.pi * x)

    def solution(self, p: TensorLike, t: float) -> TensorLike:
        x = p[..., 0]
        term1 = bm.cos(4 * bm.pi * t) * bm.sin(4 * bm.pi * x)
        term2 = (1 / (8 * bm.pi)) * bm.sin(8 * bm.pi * t) * bm.sin(8 * bm.pi * x)
        return term1 + term2

    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        x = p[..., 0]
        term1 = bm.cos(4 * bm.pi * t) * 4 * bm.pi * bm.cos(4 * bm.pi * x)
        term2 = (1 / (8 * bm.pi)) * bm.sin(8 * bm.pi * t) * 8 * bm.pi * bm.cos(8 * bm.pi * x)
        return term1 + term2  # du/dx

    def source(self, p: TensorLike, t: float) -> TensorLike:
        # Homogeneous wave equation: source term is zero
        x = p[..., 0]
        return bm.zeros_like(x)

    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        x = p[..., 0]
        return bm.zeros_like(x)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 1.0) < 1e-12)


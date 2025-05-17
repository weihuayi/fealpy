from typing import Sequence
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

    def duration(self) -> Sequence[float]:
        return [0.0, 1.4]  # Time interval [t0, t1]

    def init_solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

    def init_solution_t(self, p: TensorLike) -> TensorLike:
        # Initial velocity is zero everywhere
        return bm.zeros_like(p[..., 0])

    def solution(self, p: TensorLike, t: float) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.cos(bm.sqrt(2.0) * bm.pi * t) * bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        factor = bm.cos(bm.sqrt(2.0) * bm.pi * t)
        dx = bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y) * factor
        dy = bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y) * factor
        return bm.stack([dx, dy], axis=-1)

    def source(self, p: TensorLike, t: float) -> TensorLike:
        # Homogeneous wave equation: source term is zero
        return bm.zeros_like(p[..., 0])

    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 1.0) < 1e-12) | \
               (bm.abs(y - 0.0) < 1e-12) | (bm.abs(y - 1.0) < 1e-12)


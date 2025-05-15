from typing import Sequence
from ...backend import TensorLike
from ...backend import backend_manager as bm

class SinExpData1D:
    """
    1D parabolic problem:

        ∂u/∂t - ∂²u/∂x² = f,     x ∈ (0, 1), t > 0
        u(0, t) = u(1, t) = 0,   t > 0
        u(x, 0) = sin(4πx),      x ∈ (0, 1)

    Exact solution:

        u(x, t) = sin(4πx)·exp(-10t)
        f(x, t) = -10·exp(-10t)·sin(4πx) + 16π²·exp(-10t)·sin(4πx)

    This example imposes homogeneous Dirichlet boundary conditions at both ends.
    It is useful for testing time-dependent solvers.
    """

    def geo_dimension(self) -> int:
        return 1

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0]

    def duration(self) -> Sequence[float]:
        return [0.0, 1.0]

    def init_solution(self, p: TensorLike) -> TensorLike:
        """
        Initial condition at t = 0:
            u(x, 0) = sin(4πx)
        """
        x = p[..., 0]
        return bm.sin(4 * bm.pi * x)

    def solution(self, p: TensorLike, t: float) -> TensorLike:
        x = p[..., 0]
        return bm.sin(4 * bm.pi * x) * bm.exp(-10 * t)

    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        x = p[..., 0]
        return 4 * bm.pi * bm.cos(4 * bm.pi * x) * bm.exp(-10 * t)

    def source(self, p: TensorLike, t: float) -> TensorLike:
        x = p[..., 0]
        term1 = -10 * bm.exp(-10 * t) * bm.sin(4 * bm.pi * x)
        term2 = 16 * bm.pi ** 2 * bm.exp(-10 * t) * bm.sin(4 * bm.pi * x)
        return term1 + term2

    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        return self.solution(p, t)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return (bm.abs(x - 0.0) < 1e-12) | (bm.abs(x - 1.0) < 1e-12)


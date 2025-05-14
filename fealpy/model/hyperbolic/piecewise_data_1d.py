from typing import Sequence
from fealpy.decorator import cartesian
from ...backend import TensorLike
from ...backend import backend_manager as bm

class PiecewiseData1d:
    """
    1D hyperbolic problem with piecewise linear solution:

        ∂u/∂t + a·∂u/∂x = 0,     x ∈ (0, 2), t ∈ (0, 4)
        u(x, 0) = |x - 1|,        x ∈ (0, 2)
        u(0, t) = 1,              t ∈ (0, 4)

    Exact solution is a piecewise linear function with three regions:
        - u = 1 for x ≤ t
        - u = 1 - x + t for t < x ≤ t+1
        - u = x - t - 1 for x > t+1
    This represents a wave propagating with speed a=1.
    """

    def geo_dimension(self) -> int:
        return 1

    def domain(self) -> Sequence[float]:
        return [0.0, 2.0]

    def duration(self) -> Sequence[float]:
        return [0.0, 4.0]

    @cartesian
    def init_solution(self, p: TensorLike) -> TensorLike:
        return bm.abs(p - 1)

    def solution(self, p: TensorLike, t: float) -> TensorLike:
        val = bm.zeros_like(p)
        flag1 = p <= t
        flag2 = p > t + 1
        flag3 = ~flag1 & ~flag2
        val[flag1] = 1
        val[flag3] = 1 - p[flag3] + t
        val[flag2] = p[flag2] - t - 1
        return val

    def gradient(self, p: TensorLike, t: float) -> TensorLike:
        grad = bm.zeros_like(p)
        flag1 = p <= t
        flag2 = p > t + 1
        flag3 = ~flag1 & ~flag2
        grad[flag1] = 0
        grad[flag3] = -1
        grad[flag2] = 1
        return grad

    def source(self, p: TensorLike, t: float) -> TensorLike:
        return bm.zeros_like(p)

    def dirichlet(self, p: TensorLike, t: float) -> TensorLike:
        return bm.ones_like(p)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        return (bm.abs(p - 0.0) < 1e-12) | (bm.abs(p - 2.0) < 1e-12)
    
    def a(self) -> float:
        """
        Wave speed
        """
        return 1.0
from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike

class SinExpData1D:
    """
    1D Kolmogorov flow (parabolic PDE):

        ∂u/∂t = ν ∂²u/∂x²,  x in (0, 1), t > 0
        u(0, t) = u(1, t),  periodic boundary conditions

    with the exact solution:

        u(x, t) = sin(2πx) · exp(-4π²νt)

    The corresponding source term is:

        f(x, t) = 0  (homogeneous)

    Periodic boundary conditions are applied at both ends of the interval.
    """

    def __init__(self, nu: float = 0.01, time: float = 0.0):
        self.nu = nu
        self.time = time

    def geo_dimension(self) -> int:
        return 1

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        pi = bm.pi
        val = bm.sin(2 * pi * x) * bm.exp(-4 * pi**2 * self.nu * self.time)
        return val

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        pi = bm.pi
        val = 2 * pi * bm.cos(2 * pi * x) * bm.exp(-4 * pi**2 * self.nu * self.time)
        return val

    def source(self, p: TensorLike) -> TensorLike:
        # Homogeneous right-hand side (no forcing)
        return bm.zeros_like(p[..., 0])

    def dirichlet(self, p: TensorLike) -> TensorLike:
        # Not used if periodic BCs are applied,
        # but we return solution here for compatibility
        return self.solution(p)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        # Always return False for periodic problems
        return bm.zeros_like(p[..., 0], dtype=bm.bool_)
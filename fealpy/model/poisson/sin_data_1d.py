from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike

class SinData1D:
    """
    1D Poisson problem:

        -u''(x) = f(x),  x in (0, 1)
         u(0) = u(1) = 0

    with the exact solution:

        u(x) = sin(πx)

    The corresponding source term is:

        f(x) = π²·sin(πx)

    Dirichlet boundary conditions are applied at both ends of the interval.
    """

    def geo_dimension(self) -> int: 
        return 1

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        pi = bm.pi
        val = bm.sin(pi * x)
        return val

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        pi = bm.pi
        val = pi * bm.cos(pi * x)
        return val

    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        pi = bm.pi
        val = pi**2 * bm.sin(pi * x)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)


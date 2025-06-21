from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike

class SinData1D:
    """
    1D ABC-type problem (simplified):

        -u''(x) = f(x),  x in (0, 1)
         u(0) = u(1) = 0

    with the exact solution:

        u(x) = sin(2πx)

    The corresponding source term is:

        f(x) = (2π)^2 · sin(2πx)

    Dirichlet boundary conditions are applied at both ends.
    """

    def geo_dimension(self) -> int:
        return 1

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        val = bm.sin(2 * bm.pi * x)
        return val

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        val = 2 * bm.pi * bm.cos(2 * bm.pi * x)
        return val

    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        val = (2 * bm.pi) ** 2 * bm.sin(2 * bm.pi * x)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p
        atol = 1e-5
        return (bm.abs(x - 1.0) < atol) | (bm.abs(x) < atol)
from typing import Sequence

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

class SinSinData2D:
    """
    2D ABC-type PDE:

        -Δu(x, y) = f(x, y),   (x, y) in (0, 1)^2
         u(x, y) = 0,          on ∂Ω

    with the exact solution:

        u(x, y) = sin(2πx) * sin(2πy)

    The corresponding source term is:

        f(x, y) = 8π² · sin(2πx) · sin(2πy)

    Dirichlet boundary conditions are applied on all sides.
    """

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        val = bm.sin(2 * bm.pi * x) * bm.sin(2 * bm.pi * y)
        return val

    def gradient(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        du_dx = 2 * bm.pi * bm.cos(2 * bm.pi * x) * bm.sin(2 * bm.pi * y)
        du_dy = 2 * bm.pi * bm.sin(2 * bm.pi * x) * bm.cos(2 * bm.pi * y)
        return bm.stack([du_dx, du_dy], axis=-1)

    def source(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        val = 8 * bm.pi ** 2 * bm.sin(2 * bm.pi * x) * bm.sin(2 * bm.pi * y)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-5
        return (
            (bm.abs(x - 1.0) < atol) |
            (bm.abs(x) < atol) |
            (bm.abs(y - 1.0) < atol) |
            (bm.abs(y) < atol)
        )
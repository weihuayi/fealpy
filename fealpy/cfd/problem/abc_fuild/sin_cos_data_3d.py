from typing import Sequence
from ...backend import backend_manager as bm
from ...backend import TensorLike

class SinCosData3D:
    """
    3D ABC flow type PDE (simplified Poisson problem):

        -Δu(x,y,z) = f(x,y,z),   (x,y,z) in (0, 1)^3
         u(x,y,z) = 0,           on ∂Ω

    The exact solution is based on a combination of sine and cosine functions typical of ABC flow:

        u(x,y,z) = sin(2πx) * cos(2πy) + sin(2πy) * cos(2πz) + sin(2πz) * cos(2πx)

    The source term f is calculated accordingly as

        f = -Δu = (2π)^2 * [ 3 * u(x,y,z) ]

    Dirichlet boundary conditions are applied on the cube boundary.
    """

    def geo_dimension(self) -> int:
        return 3

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        val = (
            bm.sin(2 * bm.pi * x) * bm.cos(2 * bm.pi * y) +
            bm.sin(2 * bm.pi * y) * bm.cos(2 * bm.pi * z) +
            bm.sin(2 * bm.pi * z) * bm.cos(2 * bm.pi * x)
        )
        return val

    def gradient(self, p: TensorLike) -> TensorLike:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        du_dx = 2 * bm.pi * bm.cos(2 * bm.pi * x) * bm.cos(2 * bm.pi * y) - 2 * bm.pi * bm.sin(2 * bm.pi * z) * bm.sin(2 * bm.pi * x)
        du_dy = -2 * bm.pi * bm.sin(2 * bm.pi * x) * bm.sin(2 * bm.pi * y) + 2 * bm.pi * bm.cos(2 * bm.pi * y) * bm.cos(2 * bm.pi * z)
        du_dz = -2 * bm.pi * bm.sin(2 * bm.pi * y) * bm.sin(2 * bm.pi * z) + 2 * bm.pi * bm.cos(2 * bm.pi * z) * bm.cos(2 * bm.pi * x)
        return bm.stack([du_dx, du_dy, du_dz], axis=-1)

    def source(self, p: TensorLike) -> TensorLike:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        u = self.solution(p)
        val = 3 * (2 * bm.pi) ** 2 * u
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)

    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        atol = 1e-5
        on_boundary = (
            (bm.abs(x - 1.0) < atol) | (bm.abs(x) < atol) |
            (bm.abs(y - 1.0) < atol) | (bm.abs(y) < atol) |
            (bm.abs(z - 1.0) < atol) | (bm.abs(z) < atol)
        )
        return on_boundary
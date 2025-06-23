from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike


class CosCosCosData3D:
    """
    3D Poisson problem:

        -Δu(x, y, z) = f(x, y, z),  (x, y, z) ∈ (0, 1)^3
         u(x, y, z) = g(x, y, z),            on ∂Ω

    with the exact solution

        u(x, y, z) = cos(πx)·cos(πy)·cos(πz)

    The corresponding source term is:

        f(x, y, z) = 3·π²·cos(πx)·cos(πy)·cos(πz)

    Homogeneous Dirichlet boundary conditions are applied on all boundaries.
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        return bm.cos(pi * x) * bm.cos(pi * y) * bm.cos(pi * z)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        du_dx = - pi * bm.sin(pi * x) * bm.cos(pi * y) * bm.cos(pi * z)
        du_dy = - pi * bm.cos(pi * x) * bm.sin(pi * y) * bm.cos(pi * z)
        du_dz = - pi * bm.cos(pi * x) * bm.cos(pi * y) * bm.sin(pi * z)
        return bm.stack([du_dx, du_dy, du_dz], axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        return 3 * pi**2 * bm.cos(pi * x) * bm.cos(pi * y) * bm.cos(pi * z)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        atol = 1e-12
        return (
            (bm.abs(x - 1.0) < atol) | (bm.abs(x) < atol) |
            (bm.abs(y - 1.0) < atol) | (bm.abs(y) < atol) |
            (bm.abs(z - 1.0) < atol) | (bm.abs(z) < atol)
        )

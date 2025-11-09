from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher3d
from fealpy.backend import bm


class Exp0001(BoxMesher3d):
    """
    3D Poisson problem:

        -Δu(x, y, z) = f(x, y, z),  (x, y, z) ∈ (0, 1)^3
         u(x, y, z) = 0,            on ∂Ω

    with the exact solution

        u(x, y, z) = x^2 + y^2 - 2*z^2

    The corresponding source term is:

        f(x, y, z) = 0

    Homogeneous Dirichlet boundary conditions are applied on all boundaries.
    """
    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] 
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]

        return x * x + y * y - 2.0 * z * z

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        u_x = 2.0 * x
        u_y = 2.0 * y
        u_z = -4.0 * z
        return bm.stack([u_x, u_y, u_z], axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        return bm.zeros_like(p[..., 0])

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

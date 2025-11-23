from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import BoxMesher3d

class Exp0007(BoxMesher3d):
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
    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] 
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain (3D)."""
        return 3

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute the exact solution u(x, y, z) = cos(πx)·cos(πy)·cos(πz)."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        return bm.cos(pi * x) * bm.cos(pi * y) * bm.cos(pi * z)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute the gradient of the solution."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        du_dx = -pi * bm.sin(pi * x) * bm.cos(pi * y) * bm.cos(pi * z)
        du_dy = -pi * bm.cos(pi * x) * bm.sin(pi * y) * bm.cos(pi * z)
        du_dz = -pi * bm.cos(pi * x) * bm.cos(pi * y) * bm.sin(pi * z)
        return bm.stack([du_dx, du_dy, du_dz], axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute the source term f(x, y, z) = 3·π²·cos(πx)·cos(πy)·cos(πz)."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        return 3 * pi**2 * bm.cos(pi * x) * bm.cos(pi * y) * bm.cos(pi * z)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition, same as the exact solution."""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if points are on the Dirichlet boundary."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        atol = 1e-12
        xmin, xmax, ymin, ymax, zmin, zmax = self.box
        return (
            (bm.abs(x - xmin) < atol) | (bm.abs(x - xmax) < atol) |
            (bm.abs(y - ymin) < atol) | (bm.abs(y - ymax) < atol) |
            (bm.abs(z - zmin) < atol) | (bm.abs(z - zmax) < atol)
        )

    
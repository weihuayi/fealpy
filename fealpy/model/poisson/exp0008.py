from typing import Optional, Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...typing import TensorLike
from ...mesher import BoxMesher2d


class Exp0008(BoxMesher2d):
    """
    2D Vector Poisson problem:

        -Δu = f,  in Ω = [0,1]^2
         u = g,   on ∂Ω

    With manufactured solution:
        u₁(x, y) = sin(πx) sin(πy)
        u₂(x, y) = cos(πx) cos(πy)

    The corresponding source term is:
        f₁(x, y) = 2π² sin(πx) sin(πy)
        f₂(x, y) = 2π² cos(πx) cos(πy)

    Non-homogeneous Dirichlet boundary conditions are applied on all edges.
    """

    def __init__(self):
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution u = (u₁, u₂)."""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        u1 = bm.sin(pi * x) * bm.sin(pi * y)
        u2 = bm.cos(pi * x) * bm.cos(pi * y)
        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution ∇u: shape (..., 2, 2)."""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        du1_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du1_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        du2_dx = -pi * bm.sin(pi * x) * bm.cos(pi * y)
        du2_dy = -pi * bm.cos(pi * x) * bm.sin(pi * y)
        return bm.stack([
            bm.stack([du1_dx, du1_dy], axis=-1),
            bm.stack([du2_dx, du2_dy], axis=-1),
        ], axis=-2)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute source term f = (f₁, f₂)."""
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        f1 = 2 * pi**2 * bm.sin(pi * x) * bm.sin(pi * y)
        f2 = 2 * pi**2 * bm.cos(pi * x) * bm.cos(pi * y)
        return bm.stack([f1, f2], axis=-1)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition u = g."""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12  # Absolute tolerance
        return (bm.abs(x - 0.0) < atol) | (bm.abs(x - 1.0) < atol) | \
               (bm.abs(y - 0.0) < atol) | (bm.abs(y - 1.0) < atol)
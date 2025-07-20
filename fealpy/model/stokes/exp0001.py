from ...backend import bm
from ...decorator import cartesian
from ...typing import TensorLike
from ..mesher import BoxMesher2d
from typing import Sequence


class Exp0001(BoxMesher2d):
    """
    Analytic solution to the 2D Stokes equations:

        -μ Δu + ∇p = f  in Ω = [0,1]^2
        ∇·u = 0         in Ω
        u = g           on ∂Ω

    With manufactured solution:
        u₁(x, y) = -cos(πx) sin(πy)
        u₂(x, y) =  sin(πx) cos(πy)
        p(x, y)  =  sin(πx) sin(πy)
        f =(cos(πx)sin(πy)(π-2μπ^2),
            sin(πx)cos(πy)(π + 2μπ^2))

    The body force f is computed accordingly.
    """

    def __init__(self, option: dict = {}):
        self.box = [0, 1, 0, 1]
        self.mu = bm.tensor(option.get("mu", 1.0))
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    def viscosity(self) -> float:
        return self.mu

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        mu = self.mu
        pi = bm.pi

        fx = bm.cos(pi * x) * bm.sin(pi * y) * (pi - 2 * mu * pi ** 2)
        fy = bm.sin(pi * x) * bm.cos(pi * y) * (pi + 2 * mu * pi ** 2)

        return bm.stack([fx, fy], axis=-1)

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        u1 = -bm.cos(bm.pi * x) * bm.sin(bm.pi * y)
        u2 =  bm.sin(bm.pi * x) * bm.cos(bm.pi * y)
        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

    @cartesian
    def grad_velocity(self, p: TensorLike) -> TensorLike:
        # Returns ∇u: shape (..., 2, 2)
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        du1_dx = pi * bm.sin(pi * x) * bm.sin(pi * y)
        du1_dy = -pi * bm.cos(pi * x) * bm.cos(pi * y)
        du2_dx = pi * bm.cos(pi * x) * bm.cos(pi * y)
        du2_dy = -pi * bm.sin(pi * x) * bm.sin(pi * y)

        return bm.stack([
            bm.stack([du1_dx, du1_dy], axis=-1),
            bm.stack([du2_dx, du2_dy], axis=-1),
        ], axis=-2)

    @cartesian
    def div_velocity(self, p: TensorLike) -> TensorLike:
        # ∇·u = ∂u₁/∂x + ∂u₂/∂y ≡ 0
        return bm.zeros(p.shape[:-1], dtype=p.dtype)

    @cartesian
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        return self.velocity(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        eps = 1e-12
        return (bm.abs(x - 0.0) < eps) | (bm.abs(x - 1.0) < eps) | \
               (bm.abs(y - 0.0) < eps) | (bm.abs(y - 1.0) < eps)

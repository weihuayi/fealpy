from ...backend import bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d
from typing import Sequence


class Exp0002(BoxMesher2d):
    """
    Analytic solution to a 2D flow problem with non-zero divergence:

        -μ Δu + ∇p = f  in Ω = [0,1]^2
        u = g           on ∂Ω

    With manufactured solution:
        u₁(x, y) = x^2
        u₂(x, y) = y^2
        p(x, y)  = xy
        f = (-2 + y, -2 + x)
        ∇·u = 2x + 2y (non-zero divergence)

    The body force f is computed to satisfy -μ Δu + ∇p = f with μ = 1.
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
        fx = -2 + y
        fy = -2 + x
        return bm.stack([fx, fy], axis=-1)

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        u1 = x * x
        u2 = y * y
        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return x * y

    @cartesian
    def grad_velocity(self, p: TensorLike) -> TensorLike:
        # Returns ∇u: shape (..., 2, 2)
        x, y = p[..., 0], p[..., 1]
        du1_dx = 2 * x
        du1_dy = 0
        du2_dx = 0
        du2_dy = 2 * y
        return bm.stack([
            bm.stack([du1_dx, du1_dy], axis=-1),
            bm.stack([du2_dx, du2_dy], axis=-1),
        ], axis=-2)

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        # Returns ∇p = (∂p/∂x, ∂p/∂y): shape (..., 2)
        x, y = p[..., 0], p[..., 1]
        dp_dx = y
        dp_dy = x
        return bm.stack([dp_dx, dp_dy], axis=-1)
    
    @cartesian
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        return self.velocity(p)

    @cartesian
    def dirichlet_pressure(self, p: TensorLike) -> TensorLike:
        return self.pressure(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        eps = 1e-12
        return (bm.abs(x - 0.0) < eps) | (bm.abs(x - 1.0) < eps) | \
               (bm.abs(y - 0.0) < eps) | (bm.abs(y - 1.0) < eps)

    @cartesian
    def div_velocity(self, p: TensorLike) -> TensorLike:
        """
        Compute the divergence of the velocity field: ∇·u = ∂u₁/∂x + ∂u₂/∂y
        Returns a scalar field with shape (...): 2x + 2y
        """
        x, y = p[..., 0], p[..., 1]
        div_u = 2 * x + 2 * y
        return div_u
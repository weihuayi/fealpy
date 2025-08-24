from ...backend import bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d
from typing import Sequence

class Exp0001(BoxMesher2d):
    """
    2D Stokes equations test case:=

        -μ Δu + ∇p = f  in Ω = [0,1]^2
        ∇·u = 0         in Ω
        u = g           on ∂Ω
        ∂p/∂n = 0       on ∂Ω   (pure zero Neumann for pressure)
        ∫_Ω p = P_target

    Exact solution:
        u₁(x,y) = π sin(πx) cos(πy)
        u₂(x,y) = -π cos(πx) sin(πy)
        p(x,y)  = cos(πx) cos(πy)

    This p satisfies ∂p/∂n = 0 on all boundaries.
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

    # === exact velocity and pressure ===
    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        u1 = bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y)
        u2 = -bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y)
        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.cos(bm.pi * x) * bm.cos(bm.pi * y)

    # === gradients ===
    @cartesian
    def grad_velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        du1_dx = pi**2 * bm.cos(pi * x) * bm.cos(pi * y)
        du1_dy = -pi**2 * bm.sin(pi * x) * bm.sin(pi * y)
        du2_dx = pi**2 * bm.sin(pi * x) * bm.sin(pi * y)
        du2_dy = -pi**2 * bm.cos(pi * x) * bm.cos(pi * y)
        return bm.stack([
            bm.stack([du1_dx, du1_dy], axis=-1),
            bm.stack([du2_dx, du2_dy], axis=-1),
        ], axis=-2)

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        dp_dx = -bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y)
        dp_dy = -bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y)
        return bm.stack([dp_dx, dp_dy], axis=-1)

    # === source term ===
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """f = -μ Δu + ∇p"""
        x, y = p[..., 0], p[..., 1]
        mu = self.mu
        pi = bm.pi

        lap_u1 = -2 * pi**3 * bm.sin(pi * x) * bm.cos(pi * y)
        lap_u2 =  2 * pi**3 * bm.cos(pi * x) * bm.sin(pi * y)

        dp_dx = -pi * bm.sin(pi * x) * bm.cos(pi * y)
        dp_dy = -pi * bm.cos(pi * x) * bm.sin(pi * y)

        fx = -mu * lap_u1 + dp_dx
        fy = -mu * lap_u2 + dp_dy
        return bm.stack([fx, fy], axis=-1)

    # === velocity Dirichlet BC ===
    @cartesian
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        return self.velocity(p)

    @cartesian
    def dirichlet_velocity_u(self, p: TensorLike) -> TensorLike:
        return self.velocity_u(p)

    @cartesian
    def dirichlet_velocity_v(self, p: TensorLike) -> TensorLike:
        return self.velocity_v(p)

    # === pressure Neumann BC (zero) ===
    @cartesian
    def neumann_pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.zeros_like(x)

    @cartesian
    def neumann_pressure_correct(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.zeros_like(x)

    # === pressure integral constraint ===
    def pressure_integral_target(self) -> float:
        return 0.0

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        eps = 1e-12
        return (bm.abs(x - 0.0) < eps) | (bm.abs(x - 1.0) < eps) | \
               (bm.abs(y - 0.0) < eps) | (bm.abs(y - 1.0) < eps)

    # === split-component functions ===
    @cartesian
    def source_u(self, p: TensorLike) -> TensorLike:
        return self.source(p)[..., 0]

    @cartesian
    def source_v(self, p: TensorLike) -> TensorLike:
        return self.source(p)[..., 1]

    @cartesian
    def velocity_u(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y)

    @cartesian
    def velocity_v(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return -bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y)

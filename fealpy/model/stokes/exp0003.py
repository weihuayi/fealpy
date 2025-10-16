from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

class Exp0003(BoxMesher2d):
    """
    2D Stokes equations test case with user-specified exact solution:

        -μ Δu + ∇p = f  in Ω = [0,1]^2
        ∇·u = 0         in Ω
        u = g           on ∂Ω
        ∂p/∂n = exact   on ∂Ω   (Neumann for pressure as per exact solution)
        ∫_Ω p = 0

    Exact solution:
        u(x,y) = 10 x^2 (x-1)^2 y (y-1) (2y-1)
        v(x,y) = -10 x (x-1) (2x-1) y^2 (y-1)^2
        p(x,y) = 10 (2x-1) (2y-1)

    This satisfies ∇·u = 0.
    μ = 1.0.
    Velocity Dirichlet BC: u = exact on ∂Ω (vanishes on boundaries).
    Pressure Neumann BC: ∂p/∂n as computed from exact gradients.
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
        u1 = 10 * x**2 * (x - 1)**2 * y * (y - 1) * (2 * y - 1)
        u2 = -10 * x * (x - 1) * (2 * x - 1) * y**2 * (y - 1)**2
        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return 10 * (2 * x - 1) * (2 * y - 1)

    # === gradients ===
    @cartesian
    def grad_velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        du1_dx = 20 * x * y * (x - 1) * (2 * x - 1) * (y - 1) * (2 * y - 1)
        du1_dy = 10 * x**2 * (x - 1)**2 * (6 * y**2 - 6 * y + 1)
        dv2_dx = -10 * y**2 * (y - 1)**2 * (6 * x**2 - 6 * x + 1)
        dv2_dy = -20 * x * y * (2 * y - 1) * (x - 1) * (2 * x - 1) * (y - 1)
        return bm.stack([
            bm.stack([du1_dx, du1_dy], axis=-1),
            bm.stack([dv2_dx, dv2_dy], axis=-1),
        ], axis=-2)

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        dp_dx = 20 * (2 * y - 1)
        dp_dy = 20 * (2 * x - 1)
        return bm.stack([dp_dx, dp_dy], axis=-1)

    # === source term ===
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """f = -μ Δu + ∇p"""
        x, y = p[..., 0], p[..., 1]
        mu = self.mu
        # lap_u = 20 * (2 * y - 1) * (3 * x**2 * (x - 1)**2 + y * (y - 1) * (6 * x**2 - 6 * x + 1))
        term_lap_u1 = 3 * x**2 * (x - 1)**2
        term_lap_u2 = y * (y - 1) * (6 * x**2 - 6 * x + 1)
        lap_u = 20 * (2 * y - 1) * (term_lap_u1 + term_lap_u2)
        # lap_v = -20 * x * (x - 1) * (2 * x - 1) * (6 * y**2 - 6 * y + 1) - 60 * y**2 * (2 * x - 1) * (y - 1)**2
        term_lap_v1 = -20 * x * (x - 1) * (2 * x - 1) * (6 * y**2 - 6 * y + 1)
        term_lap_v2 = -60 * y**2 * (2 * x - 1) * (y - 1)**2
        lap_v = term_lap_v1 + term_lap_v2
        dp_dx = 20 * (2 * y - 1)
        dp_dy = 20 * (2 * x - 1)
        fx = -mu * lap_u + dp_dx
        fy = -mu * lap_v + dp_dy
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

    # === pressure Neumann BC ===
    @cartesian
    def neumann_pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        eps = 1e-12
        mask_left = bm.abs(x) < eps
        mask_right = bm.abs(x - 1.0) < eps
        mask_bottom = bm.abs(y) < eps
        mask_top = bm.abs(y - 1.0) < eps
        dp_dx = 20 * (2 * y - 1)
        dp_dy = 20 * (2 * x - 1)
        n_p = bm.zeros_like(x)
        n_p = bm.where(mask_left, -dp_dx, n_p)
        n_p = bm.where(mask_right, dp_dx, n_p)
        n_p = bm.where(mask_bottom, -dp_dy, n_p)
        n_p = bm.where(mask_top, dp_dy, n_p)
        return n_p

    @cartesian
    def neumann_pressure_correct(self, p: TensorLike) -> TensorLike:
        return self.neumann_pressure(p)

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
        return 10 * x**2 * (x - 1)**2 * y * (y - 1) * (2 * y - 1)

    @cartesian
    def velocity_v(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return -10 * x * (x - 1) * (2 * x - 1) * y**2 * (y - 1)**2
from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d


class Exp0006(BoxMesher2d):
    """
    2D steady Navier-Stokes manufactured-solution test case
    with zero pressure Neumann boundary condition.

        -μ Δu + (u·∇)u + ∇p = f  in Ω = [0,1]^2
        ∇·u = 0                 in Ω
        u = g                   on ∂Ω (Dirichlet from exact u)
        ∂p/∂n = 0               on ∂Ω (zero Neumann for pressure)
        ∫_Ω p = 0               (pressure mean zero)

    Exact manufactured solution:

        u1(x,y) = sin(2π x) sin(π y)
        u2(x,y) = 2 cos(2π x) cos(π y)

        p(x,y) = cos(2π x) cos(2π y)

    Notes:
        - The velocity field is divergence-free.
        - The pressure field satisfies zero Neumann boundary condition
          on all four sides of [0,1]^2.
        - The pressure mean is zero.
        - This case is suitable for testing low-to-moderate Reynolds
          steady Navier-Stokes solvers.
    """

    def __init__(self, option: dict | None = None):
        if option is None:
            option = {}

        self.box = [0.0, 1.0, 0.0, 1.0]
        self.mu = bm.tensor(option.get("mu", 1.0))
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    def viscosity(self) -> TensorLike:
        return self.mu

    # === exact velocity and pressure ===
    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi

        u1 = bm.sin(2 * pi * x) * bm.sin(pi * y)
        u2 = 2 * bm.cos(2 * pi * x) * bm.cos(pi * y)

        return bm.stack([u1, u2], axis=-1)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi

        return bm.cos(2 * pi * x) * bm.cos(2 * pi * y)

    # === gradients (for convenience / testing) ===
    @cartesian
    def grad_velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi

        # ∂u1/∂x, ∂u1/∂y
        du1_dx = 2 * pi * bm.cos(2 * pi * x) * bm.sin(pi * y)
        du1_dy = pi * bm.sin(2 * pi * x) * bm.cos(pi * y)

        # ∂u2/∂x, ∂u2/∂y
        du2_dx = -4 * pi * bm.sin(2 * pi * x) * bm.cos(pi * y)
        du2_dy = -2 * pi * bm.cos(2 * pi * x) * bm.sin(pi * y)

        return bm.stack([
            bm.stack([du1_dx, du1_dy], axis=-1),
            bm.stack([du2_dx, du2_dy], axis=-1),
        ], axis=-2)

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi

        dp_dx = -2 * pi * bm.sin(2 * pi * x) * bm.cos(2 * pi * y)
        dp_dy = -2 * pi * bm.cos(2 * pi * x) * bm.sin(2 * pi * y)

        return bm.stack([dp_dx, dp_dy], axis=-1)

    # === Laplacians of velocity components ===
    @cartesian
    def lap_velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi

        # Δu1 = -5π² sin(2πx) sin(πy)
        lap_u1 = -5 * pi**2 * bm.sin(2 * pi * x) * bm.sin(pi * y)

        # Δu2 = -10π² cos(2πx) cos(πy)
        lap_u2 = -10 * pi**2 * bm.cos(2 * pi * x) * bm.cos(pi * y)

        return bm.stack([lap_u1, lap_u2], axis=-1)

    # === convective term (u·∇)u ===
    @cartesian
    def convective(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi

        conv1 = pi * bm.sin(4 * pi * x)
        conv2 = -2 * pi * bm.sin(2 * pi * y)

        return bm.stack([conv1, conv2], axis=-1)

    # === source term f = -μ Δu + (u·∇)u + ∇p ===
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        mu = self.mu

        lap_u = self.lap_velocity(p)
        conv = self.convective(p)
        grad_p = self.grad_pressure(p)

        return -mu * lap_u + conv + grad_p

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
        return bm.zeros_like(p[..., 0])

    # === global pressure integral target (remove constant) ===
    def pressure_integral_target(self) -> float:
        return 0.0

    # === convenience split functions ===
    @cartesian
    def velocity_u(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi

        return bm.sin(2 * pi * x) * bm.sin(pi * y)

    @cartesian
    def velocity_v(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi

        return 2 * bm.cos(2 * pi * x) * bm.cos(pi * y)

    @cartesian
    def source_u(self, p: TensorLike) -> TensorLike:
        return self.source(p)[..., 0]

    @cartesian
    def source_v(self, p: TensorLike) -> TensorLike:
        return self.source(p)[..., 1]
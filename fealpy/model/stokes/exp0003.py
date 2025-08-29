from ...backend import bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d
from typing import Sequence

class Exp0003(BoxMesher2d):
    """
    2D steady Navier-Stokes manufactured-solution test case:

        -μ Δu + (u·∇)u + ∇p = f  in Ω = [0,1]^2
        ∇·u = 0                 in Ω
        u = g                   on ∂Ω (Dirichlet from exact u)
        ∂p/∂n = 0               on ∂Ω (zero Neumann for pressure, optional)
        ∫_Ω p = P_target

    Exact (manufactured) solution (divergence-free):
        u1(x,y) =  π sin(π x) cos(π y)
        u2(x,y) = -π cos(π x) sin(π y)
        p(x,y)  =  cos(π x) cos(π y)

    Source term:
        f = -μ Δu + (u·∇)u + ∇p
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

    # === gradients (for convenience / testing) ===
    @cartesian
    def grad_velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        # ∂u1/∂x, ∂u1/∂y
        du1_dx = pi**2 * bm.cos(pi * x) * bm.cos(pi * y)
        du1_dy = -pi**2 * bm.sin(pi * x) * bm.sin(pi * y)
        # ∂u2/∂x, ∂u2/∂y
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

    # === Laplacians of velocity components (pre-simplified) ===
    # (used to assemble -μ Δu term)
    @cartesian
    def lap_velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        # From symbolic simplification:
        # Δ u1 = -2 * pi^3 * sin(pi x) * cos(pi y)
        # Δ u2 =  2 * pi^3 * cos(pi x) * sin(pi y)
        lap_u1 = -2 * pi**3 * bm.sin(pi * x) * bm.cos(pi * y)
        lap_u2 =  2 * pi**3 * bm.cos(pi * x) * bm.sin(pi * y)
        return bm.stack([lap_u1, lap_u2], axis=-1)

    # === convective term (u·∇)u — we compute explicitly for clarity ===
    @cartesian
    def convective(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        pi = bm.pi
        # using formula simplification:
        # (u·∇)u = [ π^3/2 * sin(2π x),  π^3/2 * sin(2π y) ]
        conv1 = (pi**3 / 2.0) * bm.sin(2.0 * pi * x)
        conv2 = (pi**3 / 2.0) * bm.sin(2.0 * pi * y)
        return bm.stack([conv1, conv2], axis=-1)

    # === source term f = -μ Δu + (u·∇)u + ∇p ===
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        mu = self.mu
        pi = bm.pi

        # Laplacian terms (as above)
        lap_u1 = -2 * pi**3 * bm.sin(pi * x) * bm.cos(pi * y)
        lap_u2 =  2 * pi**3 * bm.cos(pi * x) * bm.sin(pi * y)

        # Convective terms
        conv1 = (pi**3 / 2.0) * bm.sin(2.0 * pi * x)
        conv2 = (pi**3 / 2.0) * bm.sin(2.0 * pi * y)

        # Pressure gradient
        dp_dx = -pi * bm.sin(pi * x) * bm.cos(pi * y)
        dp_dy = -pi * bm.cos(pi * x) * bm.sin(pi * y)

        fx = -mu * lap_u1 + conv1 + dp_dx
        fy = -mu * lap_u2 + conv2 + dp_dy

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

    # === global pressure integral target (remove constant) ===
    def pressure_integral_target(self) -> float:
        return 0.0

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        eps = 1e-12
        return (bm.abs(x - 0.0) < eps) | (bm.abs(x - 1.0) < eps) | \
               (bm.abs(y - 0.0) < eps) | (bm.abs(y - 1.0) < eps)

    # === split-component convenience functions ===
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
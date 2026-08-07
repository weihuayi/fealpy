from __future__ import annotations

from typing import Sequence

from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesher import CylinderMesher


class Exp0014(CylinderMesher):
    r"""Steady fully developed flow in a straight circular pipe.

    The cylinder axis is the :math:`z` direction, with radius ``R`` and
    length ``L``.  The exact velocity family is

    .. math::

        U=(0,0,w_m),\qquad
        w_m=\left(1-\frac{x^2+y^2}{R^2}\right)^m,

    where ``m=1`` is the quadratic Poiseuille profile and ``m=2`` is a
    smooth quartic sidecar used to detect quadratic-reproduction bias.  Both
    profiles have zero convection and constant pressure, so the momentum
    source is :math:`f=-\mu\Delta U`.
    """

    def __init__(self, option: dict | None = None):
        option = {} if option is None else option
        self.options = option
        self.radius = float(option.get("radius", 0.5))
        self.length = float(option.get("length", 3.0))
        self.lc = float(option.get("lc", 0.2))
        self.profile_power = int(option.get("profile_power", 1))
        self.eps = float(option.get("eps", 1.0e-10))
        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")
        if self.length <= 0.0:
            raise ValueError("length must be positive.")
        if self.lc <= 0.0:
            raise ValueError("lc must be positive.")
        if self.profile_power not in (1, 2):
            raise ValueError("profile_power must be 1 or 2.")
        self.rho = bm.tensor(option.get("rho", 1.0))
        self.mu = bm.tensor(option.get("mu", 1.0))
        self.default_mesh_type = "tet"
        self.box = [
            -self.radius,
            self.radius,
            -self.radius,
            self.radius,
            0.0,
            self.length,
        ]
        super().__init__(radius=self.radius, height=self.length, lc=self.lc)

    def geo_dimension(self) -> int:
        return 3

    def get_dimension(self) -> int:
        return 3

    def domain(self) -> Sequence[float]:
        return self.box

    def viscosity(self) -> TensorLike:
        return self.mu

    def density(self) -> TensorLike:
        return self.rho

    def radial_factor(self, p: TensorLike) -> TensorLike:
        radius_squared = p[..., 0] ** 2 + p[..., 1] ** 2
        return 1.0 - radius_squared / self.radius**2

    def axial_profile(self, p: TensorLike) -> TensorLike:
        return self.radial_factor(p) ** self.profile_power

    @cartesian
    def velocity(self, p: TensorLike, t=None) -> TensorLike:
        w = self.axial_profile(p)
        zero = bm.zeros_like(w)
        return bm.stack([zero, zero, w], axis=-1)

    @cartesian
    def velocity_u(self, p: TensorLike, t=None) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    @cartesian
    def velocity_v(self, p: TensorLike, t=None) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    @cartesian
    def velocity_w(self, p: TensorLike, t=None) -> TensorLike:
        return self.axial_profile(p)

    @cartesian
    def velocity_0(self, p: TensorLike, t0=None) -> TensorLike:
        return self.velocity(p)

    @cartesian
    def pressure(self, p: TensorLike, t=None) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    @cartesian
    def pressure_0(self, p: TensorLike, t0=None) -> TensorLike:
        return self.pressure(p)

    @cartesian
    def grad_pressure(self, p: TensorLike, t=None) -> TensorLike:
        return bm.zeros_like(p)

    @cartesian
    def grad_velocity(self, p: TensorLike, t=None) -> TensorLike:
        factor = self.radial_factor(p)
        coefficient = (
            -2.0
            * self.profile_power
            * factor ** (self.profile_power - 1)
            / self.radius**2
        )
        dw_dx = coefficient * p[..., 0]
        dw_dy = coefficient * p[..., 1]
        zero = bm.zeros_like(dw_dx)
        zero_row = bm.stack([zero, zero, zero], axis=-1)
        w_row = bm.stack([dw_dx, dw_dy, zero], axis=-1)
        return bm.stack([zero_row, zero_row, w_row], axis=-2)

    @cartesian
    def div_velocity(self, p: TensorLike, t=None) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    @cartesian
    def lap_velocity(self, p: TensorLike, t=None) -> TensorLike:
        factor = self.radial_factor(p)
        radial_squared = p[..., 0] ** 2 + p[..., 1] ** 2
        power = self.profile_power
        lap_w = -4.0 * power * factor ** (power - 1) / self.radius**2
        if power == 2:
            lap_w = lap_w + 8.0 * radial_squared / self.radius**4
        zero = bm.zeros_like(lap_w)
        return bm.stack([zero, zero, lap_w], axis=-1)

    @cartesian
    def convective(self, p: TensorLike, t=None) -> TensorLike:
        return bm.zeros_like(p)

    @cartesian
    def source(self, p: TensorLike, t=None) -> TensorLike:
        return -self.mu * self.lap_velocity(p)

    @cartesian
    def source_u(self, p: TensorLike, t=None) -> TensorLike:
        return self.source(p)[..., 0]

    @cartesian
    def source_v(self, p: TensorLike, t=None) -> TensorLike:
        return self.source(p)[..., 1]

    @cartesian
    def source_w(self, p: TensorLike, t=None) -> TensorLike:
        return self.source(p)[..., 2]

    @cartesian
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        return self.velocity(p)

    @cartesian
    def dirichlet_velocity_u(self, p: TensorLike) -> TensorLike:
        return self.velocity_u(p)

    @cartesian
    def dirichlet_velocity_v(self, p: TensorLike) -> TensorLike:
        return self.velocity_v(p)

    @cartesian
    def dirichlet_velocity_w(self, p: TensorLike) -> TensorLike:
        return self.velocity_w(p)

    @cartesian
    def dirichlet_pressure(self, p: TensorLike, t=None) -> TensorLike:
        return self.pressure(p)

    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        return bm.abs(p[..., 2]) <= self.eps

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        return bm.abs(p[..., 2] - self.length) <= self.eps

    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        radius = bm.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
        end = self.is_inlet_boundary(p) | self.is_outlet_boundary(p)
        return (~end) & (bm.abs(radius - self.radius) <= max(self.lc, self.eps))

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        return (
            self.is_inlet_boundary(p)
            | self.is_outlet_boundary(p)
            | self.is_wall_boundary(p)
        )

    is_dirichlet_boundary = is_velocity_boundary

    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None):
        if p is None:
            return 0
        return bm.zeros_like(p[..., 0], dtype=bm.bool)

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        return self.is_velocity_boundary(p)

    @cartesian
    def neumann_pressure(self, p: TensorLike) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    def pressure_integral_target(self) -> float:
        return 0.0

    def exact_flow_rate(self) -> float:
        return float(bm.pi) * self.radius**2 / (self.profile_power + 1)

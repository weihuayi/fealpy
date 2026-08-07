from math import pi
from typing import Sequence

from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesher import BoxMesher2d


class Exp0011(BoxMesher2d):
    """Smooth unsteady manufactured solution on the unit square.

    The velocity is generated from a stream function

        psi = sin(pi*x)^2 sin(pi*y)^2 cos(t),

    so the exact velocity is divergence-free and vanishes on the whole
    boundary for all time.  The pressure has zero spatial mean:

        p = cos(2*pi*x) cos(2*pi*y) sin(t).

    The source term is evaluated from

        f = du/dt + (u . grad)u + grad(p) - mu * Delta(u),

    with rho = mu = 1.  This case is intended for first-order-in-time PISO
    verification on collocated finite-volume meshes.
    """

    def __init__(self, options: dict | None = None):
        options = {} if options is None else options
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = float(options.get("eps", 1.0e-12))
        self.rho = bm.tensor(options.get("rho", 1.0))
        self.mu = bm.tensor(options.get("mu", 1.0))
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        return 2

    def get_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    def viscosity(self) -> TensorLike:
        return self.mu

    def trigonometric_data(self, p: TensorLike, t):
        x, y = p[..., 0], p[..., 1]
        sx = bm.sin(pi * x)
        cx = bm.cos(pi * x)
        sy = bm.sin(pi * y)
        cy = bm.cos(pi * y)
        ct = bm.cos(t)
        st = bm.sin(t)
        return sx, cx, sy, cy, ct, st

    @cartesian
    def velocity(self, p: TensorLike, t) -> TensorLike:
        return bm.stack([self.velocity_u(p, t), self.velocity_v(p, t)], axis=-1)

    @cartesian
    def velocity_u(self, p: TensorLike, t) -> TensorLike:
        sx, _, sy, cy, ct, _ = self.trigonometric_data(p, t)
        return 2.0 * pi * sx**2 * sy * cy * ct

    @cartesian
    def velocity_v(self, p: TensorLike, t) -> TensorLike:
        sx, cx, sy, _, ct, _ = self.trigonometric_data(p, t)
        return -2.0 * pi * sx * cx * sy**2 * ct

    @cartesian
    def velocity_0(self, p: TensorLike, t0) -> TensorLike:
        return self.velocity(p, t0)

    @cartesian
    def velocity_u0(self, p: TensorLike, t0) -> TensorLike:
        return self.velocity_u(p, t0)

    @cartesian
    def velocity_v0(self, p: TensorLike, t0) -> TensorLike:
        return self.velocity_v(p, t0)

    @cartesian
    def pressure(self, p: TensorLike, t) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.cos(2.0 * pi * x) * bm.cos(2.0 * pi * y) * bm.sin(t)

    @cartesian
    def pressure_0(self, p: TensorLike, t0) -> TensorLike:
        return self.pressure(p, t0)

    @cartesian
    def grad_pressure(self, p: TensorLike, t) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        dp_dx = -2.0 * pi * bm.sin(2.0 * pi * x) * bm.cos(2.0 * pi * y) * bm.sin(t)
        dp_dy = -2.0 * pi * bm.cos(2.0 * pi * x) * bm.sin(2.0 * pi * y) * bm.sin(t)
        return bm.stack([dp_dx, dp_dy], axis=-1)

    @cartesian
    def grad_velocity(self, p: TensorLike, t) -> TensorLike:
        sx, cx, sy, cy, ct, _ = self.trigonometric_data(p, t)
        du_dx = 4.0 * pi**2 * sx * cx * sy * cy * ct
        du_dy = 2.0 * pi**2 * sx**2 * (cy**2 - sy**2) * ct
        dv_dx = -2.0 * pi**2 * (cx**2 - sx**2) * sy**2 * ct
        dv_dy = -4.0 * pi**2 * sx * cx * sy * cy * ct
        grad_u = bm.stack([du_dx, du_dy], axis=-1)
        grad_v = bm.stack([dv_dx, dv_dy], axis=-1)
        return bm.stack([grad_u, grad_v], axis=-2)

    @cartesian
    def div_velocity(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        return bm.zeros_like(x)

    @cartesian
    def lap_velocity(self, p: TensorLike, t) -> TensorLike:
        sx, cx, sy, cy, ct, _ = self.trigonometric_data(p, t)
        lap_u = 4.0 * pi**3 * sy * cy * (cx**2 - 3.0 * sx**2) * ct
        lap_v = 4.0 * pi**3 * sx * cx * (3.0 * sy**2 - cy**2) * ct
        return bm.stack([lap_u, lap_v], axis=-1)

    @cartesian
    def time_derivative_velocity(self, p: TensorLike, t) -> TensorLike:
        sx, cx, sy, cy, _, st = self.trigonometric_data(p, t)
        ut = -2.0 * pi * sx**2 * sy * cy * st
        vt = 2.0 * pi * sx * cx * sy**2 * st
        return bm.stack([ut, vt], axis=-1)

    @cartesian
    def convective(self, p: TensorLike, t) -> TensorLike:
        u = self.velocity_u(p, t)
        v = self.velocity_v(p, t)
        grad = self.grad_velocity(p, t)
        conv_u = u * grad[..., 0, 0] + v * grad[..., 0, 1]
        conv_v = u * grad[..., 1, 0] + v * grad[..., 1, 1]
        return bm.stack([conv_u, conv_v], axis=-1)

    @cartesian
    def source(self, p: TensorLike, t) -> TensorLike:
        return (
            self.time_derivative_velocity(p, t)
            + self.convective(p, t)
            + self.grad_pressure(p, t)
            - self.mu * self.lap_velocity(p, t)
        )

    @cartesian
    def source_u(self, p: TensorLike, t) -> TensorLike:
        return self.source(p, t)[..., 0]

    @cartesian
    def source_v(self, p: TensorLike, t) -> TensorLike:
        return self.source(p, t)[..., 1]

    @cartesian
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        return bm.zeros(p.shape, dtype=getattr(p, "dtype", bm.float64))

    @cartesian
    def dirichlet_velocity_u(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return bm.zeros_like(x)

    @cartesian
    def dirichlet_velocity_v(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return bm.zeros_like(x)

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return (
            (bm.abs(x - self.box[0]) < self.eps)
            | (bm.abs(x - self.box[1]) < self.eps)
            | (bm.abs(y - self.box[2]) < self.eps)
            | (bm.abs(y - self.box[3]) < self.eps)
        )

    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None):
        return 0

    @cartesian
    def dirichlet_pressure(self, p: TensorLike, t=None):
        return None

    def pressure_integral_target(self) -> float:
        return 0.0

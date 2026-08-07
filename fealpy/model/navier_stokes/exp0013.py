from typing import Sequence

from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

from .exp0012 import Exp0012


class Exp0013(Exp0012):
    r"""Unsteady 3D box manufactured solution for PISO verification.

    The spatial velocity and pressure are inherited from :class:`Exp0012`:

    .. math::

        U(x, t) = a(t) U_0(x), \qquad p(x, t) = a(t) p_0(x),

    with ``a(t) = 1 + t``.  The velocity remains divergence-free for all
    times and vanishes on the whole boundary of the unit cube, so the current
    time-independent boundary callback used by the FVM adapters is still
    exact on boundary faces.

    The source term includes the transient contribution

    .. math::

        \rho \partial_t U + \rho (U\cdot\nabla)U + \nabla p - \mu \Delta U.
    """

    def __init__(self, option: dict | None = None):
        super().__init__(option)
        self.default_mesh_type = "uniform_hex"

    def time_factor(self, t=None):
        return 1.0 if t is None else 1.0 + t

    def time_factor_derivative(self, t=None):
        return 1.0

    def domain(self) -> Sequence[float]:
        return self.box

    @cartesian
    def base_velocity(self, p: TensorLike) -> TensorLike:
        sx, _, sy, _, sz, _, s2x, _, s2y, _, _, _ = self.trigonometric_data(p)
        u = sx**2 * s2y * sz**2
        v = -s2x * sy**2 * sz**2
        return bm.stack([u, v, bm.zeros_like(u)], axis=-1)

    @cartesian
    def base_pressure(self, p: TensorLike) -> TensorLike:
        sx, _, _, cy, sz, _, _, _, _, _, _, _ = self.trigonometric_data(p)
        return sx * cy * sz

    @cartesian
    def base_grad_pressure(self, p: TensorLike) -> TensorLike:
        sx, cx, sy, cy, sz, cz, _, _, _, _, _, _ = self.trigonometric_data(p)
        pi = bm.pi
        return bm.stack(
            [
                pi * cx * cy * sz,
                -pi * sx * sy * sz,
                pi * sx * cy * cz,
            ],
            axis=-1,
        )

    @cartesian
    def base_grad_velocity(self, p: TensorLike) -> TensorLike:
        sx, _, sy, _, sz, _, s2x, c2x, s2y, c2y, s2z, _ = self.trigonometric_data(p)
        pi = bm.pi
        du_dx = pi * s2x * s2y * sz**2
        du_dy = 2.0 * pi * sx**2 * c2y * sz**2
        du_dz = pi * sx**2 * s2y * s2z
        dv_dx = -2.0 * pi * c2x * sy**2 * sz**2
        dv_dy = -pi * s2x * s2y * sz**2
        dv_dz = -pi * s2x * sy**2 * s2z
        zero = bm.zeros_like(p[..., 0])
        return bm.stack(
            [
                bm.stack([du_dx, du_dy, du_dz], axis=-1),
                bm.stack([dv_dx, dv_dy, dv_dz], axis=-1),
                bm.stack([zero, zero, zero], axis=-1),
            ],
            axis=-2,
        )

    @cartesian
    def base_lap_velocity(self, p: TensorLike) -> TensorLike:
        sx, _, sy, _, sz, _, s2x, c2x, s2y, c2y, _, c2z = self.trigonometric_data(p)
        pi2 = bm.pi**2
        lap_u = (
            2.0 * pi2 * c2x * s2y * sz**2
            - 4.0 * pi2 * sx**2 * s2y * sz**2
            + 2.0 * pi2 * sx**2 * s2y * c2z
        )
        lap_v = (
            4.0 * pi2 * s2x * sy**2 * sz**2
            - 2.0 * pi2 * s2x * c2y * sz**2
            - 2.0 * pi2 * s2x * sy**2 * c2z
        )
        return bm.stack([lap_u, lap_v, bm.zeros_like(lap_u)], axis=-1)

    @cartesian
    def base_convective(self, p: TensorLike) -> TensorLike:
        velocity = self.base_velocity(p)
        grad = self.base_grad_velocity(p)
        conv_u = velocity[..., 0] * grad[..., 0, 0] + velocity[..., 1] * grad[..., 0, 1]
        conv_v = velocity[..., 0] * grad[..., 1, 0] + velocity[..., 1] * grad[..., 1, 1]
        return bm.stack([conv_u, conv_v, bm.zeros_like(conv_u)], axis=-1)

    @cartesian
    def velocity(self, p: TensorLike, t=None) -> TensorLike:
        return self.time_factor(t) * self.base_velocity(p)

    @cartesian
    def velocity_u(self, p: TensorLike, t=None) -> TensorLike:
        return self.velocity(p, t)[..., 0]

    @cartesian
    def velocity_v(self, p: TensorLike, t=None) -> TensorLike:
        return self.velocity(p, t)[..., 1]

    @cartesian
    def velocity_w(self, p: TensorLike, t=None) -> TensorLike:
        return self.velocity(p, t)[..., 2]

    @cartesian
    def velocity_0(self, p: TensorLike, t0=None) -> TensorLike:
        return self.velocity(p, t0)

    @cartesian
    def pressure(self, p: TensorLike, t=None) -> TensorLike:
        return self.time_factor(t) * self.base_pressure(p)

    @cartesian
    def pressure_0(self, p: TensorLike, t0=None) -> TensorLike:
        return self.pressure(p, t0)

    @cartesian
    def grad_pressure(self, p: TensorLike, t=None) -> TensorLike:
        return self.time_factor(t) * self.base_grad_pressure(p)

    @cartesian
    def grad_velocity(self, p: TensorLike, t=None) -> TensorLike:
        return self.time_factor(t) * self.base_grad_velocity(p)

    @cartesian
    def div_velocity(self, p: TensorLike, t=None) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    @cartesian
    def lap_velocity(self, p: TensorLike, t=None) -> TensorLike:
        return self.time_factor(t) * self.base_lap_velocity(p)

    @cartesian
    def convective(self, p: TensorLike, t=None) -> TensorLike:
        factor = self.time_factor(t)
        return factor * factor * self.base_convective(p)

    @cartesian
    def source(self, p: TensorLike, t=None) -> TensorLike:
        factor = self.time_factor(t)
        return (
            self.rho * self.time_factor_derivative(t) * self.base_velocity(p)
            + self.rho * factor * factor * self.base_convective(p)
            + factor * self.base_grad_pressure(p)
            - self.mu * factor * self.base_lap_velocity(p)
        )

    @cartesian
    def source_u(self, p: TensorLike, t=None) -> TensorLike:
        return self.source(p, t)[..., 0]

    @cartesian
    def source_v(self, p: TensorLike, t=None) -> TensorLike:
        return self.source(p, t)[..., 1]

    @cartesian
    def source_w(self, p: TensorLike, t=None) -> TensorLike:
        return self.source(p, t)[..., 2]

    @cartesian
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        return self.base_velocity(p)

    @cartesian
    def dirichlet_velocity_u(self, p: TensorLike) -> TensorLike:
        return self.base_velocity(p)[..., 0]

    @cartesian
    def dirichlet_velocity_v(self, p: TensorLike) -> TensorLike:
        return self.base_velocity(p)[..., 1]

    @cartesian
    def dirichlet_velocity_w(self, p: TensorLike) -> TensorLike:
        return self.base_velocity(p)[..., 2]

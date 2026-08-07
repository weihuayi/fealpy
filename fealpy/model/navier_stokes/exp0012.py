from typing import Sequence

from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesher import BoxMesher3d


class Exp0012(BoxMesher3d):
    r"""Steady 3D box manufactured solution with non-zero pressure.

    The domain is the unit cube.  The exact velocity is divergence-free and
    vanishes on the whole boundary:

    .. math::

        u_1 = \sin^2(\pi x)\sin(2\pi y)\sin^2(\pi z),

        u_2 = -\sin(2\pi x)\sin^2(\pi y)\sin^2(\pi z),

        u_3 = 0.

    The pressure is non-zero and has zero spatial mean:

    .. math::

        p = \sin(\pi x)\cos(\pi y)\sin(\pi z).

    This case is intended as a clean 3D finite-volume pressure benchmark:
    the geometry has only planar boundaries, all velocity boundary values are
    strict Dirichlet data, and pressure is fixed by a zero-mean gauge.
    """

    def __init__(self, option: dict | None = None):
        option = {} if option is None else option
        self.options = option
        self.box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        self.eps = float(option.get("eps", 1.0e-12))
        self.rho = bm.tensor(option.get("rho", 1.0))
        self.mu = bm.tensor(option.get("mu", 1.0))
        self.default_mesh_type = "uniform_tet"
        super().__init__(box=self.box)

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

    def trigonometric_data(self, p: TensorLike):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        pi = bm.pi
        sx, cx = bm.sin(pi * x), bm.cos(pi * x)
        sy, cy = bm.sin(pi * y), bm.cos(pi * y)
        sz, cz = bm.sin(pi * z), bm.cos(pi * z)
        s2x, c2x = bm.sin(2.0 * pi * x), bm.cos(2.0 * pi * x)
        s2y, c2y = bm.sin(2.0 * pi * y), bm.cos(2.0 * pi * y)
        s2z, c2z = bm.sin(2.0 * pi * z), bm.cos(2.0 * pi * z)
        return sx, cx, sy, cy, sz, cz, s2x, c2x, s2y, c2y, s2z, c2z

    @cartesian
    def velocity(self, p: TensorLike, t=None) -> TensorLike:
        return bm.stack(
            [
                self.velocity_u(p, t),
                self.velocity_v(p, t),
                self.velocity_w(p, t),
            ],
            axis=-1,
        )

    @cartesian
    def velocity_u(self, p: TensorLike, t=None) -> TensorLike:
        sx, _, _, _, sz, _, _, _, s2y, _, _, _ = self.trigonometric_data(p)
        return sx**2 * s2y * sz**2

    @cartesian
    def velocity_v(self, p: TensorLike, t=None) -> TensorLike:
        _, _, sy, _, sz, _, s2x, _, _, _, _, _ = self.trigonometric_data(p)
        return -s2x * sy**2 * sz**2

    @cartesian
    def velocity_w(self, p: TensorLike, t=None) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    @cartesian
    def velocity_0(self, p: TensorLike, t0=None) -> TensorLike:
        return self.velocity(p)

    @cartesian
    def pressure(self, p: TensorLike, t=None) -> TensorLike:
        sx, _, _, cy, sz, _, _, _, _, _, _, _ = self.trigonometric_data(p)
        return sx * cy * sz

    @cartesian
    def pressure_0(self, p: TensorLike, t0=None) -> TensorLike:
        return self.pressure(p)

    @cartesian
    def grad_pressure(self, p: TensorLike, t=None) -> TensorLike:
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
    def grad_velocity(self, p: TensorLike, t=None) -> TensorLike:
        sx, _, sy, _, sz, _, s2x, c2x, s2y, c2y, s2z, _ = self.trigonometric_data(p)
        pi = bm.pi
        du_dx = pi * s2x * s2y * sz**2
        du_dy = 2.0 * pi * sx**2 * c2y * sz**2
        du_dz = pi * sx**2 * s2y * s2z
        dv_dx = -2.0 * pi * c2x * sy**2 * sz**2
        dv_dy = -pi * s2x * s2y * sz**2
        dv_dz = -pi * s2x * sy**2 * s2z
        zero = bm.zeros_like(p[..., 0])
        grad_u = bm.stack([du_dx, du_dy, du_dz], axis=-1)
        grad_v = bm.stack([dv_dx, dv_dy, dv_dz], axis=-1)
        grad_w = bm.stack([zero, zero, zero], axis=-1)
        return bm.stack([grad_u, grad_v, grad_w], axis=-2)

    @cartesian
    def div_velocity(self, p: TensorLike, t=None) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    @cartesian
    def lap_velocity(self, p: TensorLike, t=None) -> TensorLike:
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
    def convective(self, p: TensorLike, t=None) -> TensorLike:
        u = self.velocity_u(p)
        v = self.velocity_v(p)
        grad = self.grad_velocity(p)
        conv_u = u * grad[..., 0, 0] + v * grad[..., 0, 1]
        conv_v = u * grad[..., 1, 0] + v * grad[..., 1, 1]
        return bm.stack([conv_u, conv_v, bm.zeros_like(conv_u)], axis=-1)

    @cartesian
    def source(self, p: TensorLike, t=None) -> TensorLike:
        return (
            self.rho * self.convective(p)
            + self.grad_pressure(p)
            - self.mu * self.lap_velocity(p)
        )

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
    def dirichlet_pressure(self, p: TensorLike, t=None):
        return self.pressure(p)

    @cartesian
    def neumann_pressure(self, p: TensorLike) -> TensorLike:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        grad_p = self.grad_pressure(p)
        nx = bm.where(
            bm.abs(x - self.box[0]) < self.eps,
            -1.0,
            bm.where(bm.abs(x - self.box[1]) < self.eps, 1.0, 0.0),
        )
        ny = bm.where(
            bm.abs(y - self.box[2]) < self.eps,
            -1.0,
            bm.where(bm.abs(y - self.box[3]) < self.eps, 1.0, 0.0),
        )
        nz = bm.where(
            bm.abs(z - self.box[4]) < self.eps,
            -1.0,
            bm.where(bm.abs(z - self.box[5]) < self.eps, 1.0, 0.0),
        )
        return grad_p[..., 0] * nx + grad_p[..., 1] * ny + grad_p[..., 2] * nz

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return (
            (bm.abs(x - self.box[0]) < self.eps)
            | (bm.abs(x - self.box[1]) < self.eps)
            | (bm.abs(y - self.box[2]) < self.eps)
            | (bm.abs(y - self.box[3]) < self.eps)
            | (bm.abs(z - self.box[4]) < self.eps)
            | (bm.abs(z - self.box[5]) < self.eps)
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

    def pressure_integral_target(self) -> float:
        return 0.0

from typing import Sequence

from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesher import BoxMesher2d


class Exp0010(BoxMesher2d):
    """Steady 2D Poiseuille flow in the unit channel.

    This is the steady finite-volume version of the CFD module's
    ``Poiseuille2D`` data.  The velocity is the classical parabolic channel
    profile and the pressure is shifted to zero mean, which is physically
    equivalent because incompressible pressure is determined up to a constant.

        u = (4 y (1-y), 0),  p = mu * (4 - 8 x),  Omega = [0, 1]^2.

    With this pressure scaling, ``-mu * Delta u + grad(p) = 0`` and the
    convective term vanishes, so the body source is zero for any positive
    viscosity ``mu``.
    """

    def __init__(self, option: dict | None = None):
        option = {} if option is None else option
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = float(option.get("eps", 1.0e-12))
        self.rho = bm.tensor(option.get("rho", 1.0))
        self.mu = bm.tensor(option.get("mu", 1.0))
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    def viscosity(self) -> TensorLike:
        return self.mu

    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        y = p[..., 1]
        u = 4.0 * y * (1.0 - y)
        v = bm.zeros_like(y)
        return bm.stack([u, v], axis=-1)

    @cartesian
    def velocity_u(self, p: TensorLike) -> TensorLike:
        y = p[..., 1]
        return 4.0 * y * (1.0 - y)

    @cartesian
    def velocity_v(self, p: TensorLike) -> TensorLike:
        y = p[..., 1]
        return bm.zeros_like(y)

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return self.mu * (4.0 - 8.0 * x)

    @cartesian
    def dirichlet_pressure(self, p: TensorLike) -> TensorLike:
        return self.pressure(p)

    @cartesian
    def grad_pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        dp_dx = -8.0 * self.mu * bm.ones_like(x)
        dp_dy = bm.zeros_like(x)
        return bm.stack([dp_dx, dp_dy], axis=-1)

    @cartesian
    def grad_velocity(self, p: TensorLike) -> TensorLike:
        y = p[..., 1]
        zero = bm.zeros_like(y)
        du_dy = 4.0 - 8.0 * y
        grad_u = bm.stack([zero, du_dy], axis=-1)
        grad_v = bm.stack([zero, zero], axis=-1)
        return bm.stack([grad_u, grad_v], axis=-2)

    @cartesian
    def div_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return bm.zeros_like(x)

    @cartesian
    def lap_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        lap_u = -8.0 * bm.ones_like(x)
        lap_v = bm.zeros_like(x)
        return bm.stack([lap_u, lap_v], axis=-1)

    @cartesian
    def convective(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        zero = bm.zeros_like(x)
        return bm.stack([zero, zero], axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        zero = bm.zeros_like(x)
        return bm.stack([zero, zero], axis=-1)

    @cartesian
    def source_u(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return bm.zeros_like(x)

    @cartesian
    def source_v(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return bm.zeros_like(x)

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
    def neumann_pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        value = bm.zeros_like(x)
        left = bm.abs(x - self.box[0]) < self.eps
        right = bm.abs(x - self.box[1]) < self.eps
        value[left] = 8.0 * self.mu
        value[right] = -8.0 * self.mu
        return value

    @cartesian
    def neumann_pressure_correct(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return bm.zeros_like(x)

    def pressure_integral_target(self) -> float:
        return 0.0

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
    def is_pressure_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return (
            (bm.abs(x - self.box[0]) < self.eps)
            | (bm.abs(x - self.box[1]) < self.eps)
        )

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        return self.is_velocity_boundary(p)

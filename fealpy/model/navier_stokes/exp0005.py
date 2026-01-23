from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d


class Exp0005(BoxMesher2d):
    """
    2D steady incompressible Navier–Stokes equations
    Lid-driven cavity flow.

    Domain:
        Ω = [0, 1] × [0, 1]

    Governing equations:
        (u · ∇) u - (1/Re) Δu + ∇p = 0   in Ω
        ∇ · u = 0                      in Ω

    Boundary conditions:
        u = (1, 0)   on y = 1  (moving lid)
        u = (0, 0)   on y = 0, x = 0, x = 1 (no-slip walls)

        Pressure:
            ∫_Ω p dx = 0   (remove null space)
    """

    def __init__(self, option: dict = {}):
        self.box = [0.0, 1.0, 0.0, 1.0]

        # Reynolds number
        self.Re = bm.tensor(option.get("Re", 1.0), dtype=bm.float64)

        super().__init__(box=self.box)

    # === geometry ===
    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    # === viscosity ===
    def viscosity(self) -> float:
        return 1.0 / self.Re

    # === source term (zero) ===
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        return bm.zeros_like(p)

    @cartesian
    def source_u(self, p: TensorLike) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    @cartesian
    def source_v(self, p: TensorLike) -> TensorLike:
        return bm.zeros_like(p[..., 0])

    # === velocity Dirichlet boundary ===
    @cartesian
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        eps = 1e-12

        u = bm.zeros_like(x)
        v = bm.zeros_like(x)

        # top lid: y = 1
        lid = bm.abs(y - 1.0) < eps
        u = bm.where(lid, bm.ones_like(u), u)
        v = bm.where(lid, bm.zeros_like(v), v)

        return bm.stack([u, v], axis=-1)

    @cartesian
    def dirichlet_velocity_u(self, p: TensorLike) -> TensorLike:
        return self.dirichlet_velocity(p)[..., 0]

    @cartesian
    def dirichlet_velocity_v(self, p: TensorLike) -> TensorLike:
        return self.dirichlet_velocity(p)[..., 1]

    # === pressure Neumann BC (zero) ===
    @cartesian
    def neumann_pressure(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.zeros_like(x)

    @cartesian
    def neumann_pressure_correct(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return bm.zeros_like(x)

    # === pressure constraint ===
    def pressure_integral_target(self) -> float:
        return 0.0

    # === boundary indicator ===
    @cartesian
    def is_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        eps = 1e-12
        return (bm.abs(x - 0.0) < eps) | (bm.abs(x - 1.0) < eps) | \
               (bm.abs(y - 0.0) < eps) | (bm.abs(y - 1.0) < eps)

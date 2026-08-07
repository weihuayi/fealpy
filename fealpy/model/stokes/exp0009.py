from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace


def _build_inflow_profile(box: tuple[float, float, float, float]):
    xmin, xmax, ymin, ymax = map(float, box)

    @cartesian
    def inflow(points: Any) -> Any:
        coords = bm.asarray(points, dtype=float)
        values = bm.zeros_like(coords, dtype=float)
        tol = max(1.0e-4, 1.0e-6 * max(abs(xmin), abs(xmax), abs(ymin), abs(ymax), 1.0))
        mask = bm.abs(coords[..., 0] - xmin) <= tol
        width = float(ymax - ymin)
        if width <= 0.0:
            raise ValueError("box height must be positive")
        eta = (coords[..., 1] - ymin) / width
        values[..., 0] = bm.where(mask, 4.0 * eta * (1.0 - eta), 0.0)
        return values

    return inflow


@dataclass(slots=True)
class ObstacleStokesFluidModel:
    """Obstacle-in-Stokes model with thin FEALPy solver hooks."""

    box: tuple[float, float, float, float]
    viscosity: float = 1.0
    body_force: Any = 0.0
    pressure_neumann: bool = True
    pressure_integral_target: float = 0.0
    dirichlet_velocity_data: Any = field(init=False)
    dirichlet_velocity_threshold: Any = None
    dirichlet_pressure_data: Any = 0.0
    dirichlet_pressure_threshold: Any = None
    velocity_space: Any = None
    pressure_space: Any = None
    mesh: Any = None

    def __post_init__(self) -> None:
        self.dirichlet_velocity_data = _build_inflow_profile(self.box)

    def state_spaces(self, mesh: Any, geometry_contract: Any = None) -> tuple[Any, Any]:
        scalar_velocity_space = LagrangeFESpace(mesh, p=2)
        velocity_space = TensorFunctionSpace(scalar_velocity_space, (2, -1))
        pressure_space = LagrangeFESpace(mesh, p=1)
        self.mesh = mesh
        self.velocity_space = velocity_space
        self.pressure_space = pressure_space
        self.dirichlet_velocity_threshold = self.is_velocity_boundary
        self.dirichlet_pressure_threshold = self.is_pressure_boundary
        return velocity_space, pressure_space

    def build_state_spaces(self, mesh: Any, geometry_contract: Any = None) -> tuple[Any, Any]:
        return self.state_spaces(mesh, geometry_contract)

    @cartesian
    def is_velocity_boundary(self, p: Any = None) -> Any:
        if p is None:
            return None
        coords = bm.asarray(p, dtype=float)
        xmax = float(self.box[1])
        tol = max(1.0e-4, 1.0e-6 * max(abs(xmax), 1.0))
        return bm.abs(coords[..., 0] - xmax) > tol

    @cartesian
    def is_pressure_boundary(self, p: Any = None) -> Any:
        if p is None:
            return 0 if bool(self.pressure_neumann) else 1
        coords = bm.asarray(p, dtype=float)
        return bm.zeros(coords[..., 0].shape, dtype=bool)

    @cartesian
    def dirichlet_velocity(self, p: Any) -> Any:
        return _build_inflow_profile(self.box)(p)

    @cartesian
    def dirichlet_pressure(self, p: Any) -> Any:
        coords = bm.asarray(p, dtype=float)
        return bm.zeros(coords[..., 0].shape, dtype=float)

    @cartesian
    def source(self, p: Any) -> Any:
        coords = bm.asarray(p, dtype=float)
        return bm.zeros_like(coords, dtype=float)


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace


def _boundary_entities_to_edges(mesh: Any, boundary_entities: Any) -> Any:
    entities = bm.asarray(boundary_entities, dtype=int)
    edge = bm.asarray(mesh.entity("edge"), dtype=int)
    if entities.size == 0:
        return bm.empty(0, dtype=int)
    if entities.ndim == 2:
        edge_lookup = {tuple(sorted(map(int, edge_nodes[:2]))): index for index, edge_nodes in enumerate(edge)}
        edge_ids = [edge_lookup[tuple(sorted(map(int, entity[:2])))] for entity in entities if tuple(sorted(map(int, entity[:2]))) in edge_lookup]
        return bm.asarray(edge_ids, dtype=int)
    if entities.ndim != 1:
        raise ValueError("boundary entities must be a node id list or an edge list")
    node_mask = bm.zeros(mesh.number_of_nodes(), dtype=bool)
    node_mask[entities] = True
    return bm.flatnonzero(bm.all(node_mask[edge], axis=1)).astype(int, copy=False)


def _build_elbow_inflow_profile(inlet_x: float, y_lower: float, y_upper: float, max_velocity: float):
    def inflow(points: Any) -> Any:
        coords = bm.asarray(points, dtype=float)
        values = bm.zeros_like(coords, dtype=float)
        tol_x = max(1.0e-4, 1.0e-6 * max(abs(float(inlet_x)), abs(float(y_lower)), abs(float(y_upper)), 1.0))
        tol_y = max(1.0e-4, 1.0e-6 * max(abs(float(y_upper - y_lower)), 1.0))
        mask = (bm.abs(coords[..., 0] - inlet_x) <= tol_x) & (
            (coords[..., 1] >= y_lower - tol_y) & (coords[..., 1] <= y_upper + tol_y)
        )
        width = float(y_upper - y_lower)
        if width <= 0.0:
            raise ValueError("inlet channel width must be positive")
        eta = (coords[..., 1] - y_lower) / width
        values[..., 0] = bm.where(mask, 4.0 * float(max_velocity) * eta * (1.0 - eta), 0.0)
        values[..., 1] = 0.0
        return values

    return inflow


@dataclass(slots=True)
class ElbowPipeStokesFluidModel:
    """Paper-style 2D pipe steady NS state model."""

    inlet_x: float
    inlet_ymin: float
    inlet_ymax: float
    outlet_x: float
    rho: float = 1.0
    viscosity: float = 1.0
    inlet_max_velocity: float = 1.5
    mu: float = field(init=False, repr=False)
    body_force: Any = 0.0
    pressure_neumann: bool = True
    pressure_integral_target_value: float = 0.0
    state_system_is_linear: bool = False
    dirichlet_velocity_data: Any = field(init=False)
    dirichlet_velocity_threshold: Any = None
    dirichlet_pressure_data: Any = 0.0
    dirichlet_pressure_threshold: Any = None
    mesh: Any = None
    velocity_space: Any = None
    pressure_space: Any = None
    uspace: Any = None
    pspace: Any = None

    def __post_init__(self) -> None:
        self.rho = float(self.rho)
        self.viscosity = float(self.viscosity)
        self.mu = float(self.viscosity)
        self.inlet_max_velocity = float(self.inlet_max_velocity)
        self.dirichlet_velocity_data = _build_elbow_inflow_profile(
            self.inlet_x,
            self.inlet_ymin,
            self.inlet_ymax,
            self.inlet_max_velocity,
        )

    def state_spaces(self, mesh: Any, geometry_contract: Any = None) -> tuple[Any, Any]:
        self.mesh = mesh
        scalar_velocity_space = LagrangeFESpace(mesh, p=2)
        velocity_space = TensorFunctionSpace(scalar_velocity_space, (2, -1))
        pressure_space = LagrangeFESpace(mesh, p=1)
        self.velocity_space = velocity_space
        self.pressure_space = pressure_space
        self.uspace = velocity_space
        self.pspace = pressure_space
        self.dirichlet_velocity_threshold = self.is_velocity_boundary(velocity_space)
        self.dirichlet_pressure_threshold = self.is_pressure_boundary(pressure_space)
        return velocity_space, pressure_space

    def build_state_spaces(self, mesh: Any, geometry_contract: Any = None) -> tuple[Any, Any]:
        return self.state_spaces(mesh, geometry_contract)

    def _boundary_dofs(self, space: Any, role: str) -> Any:
        entity_names = {
            "wall": ("wall", "fsi", "design", "wall_nodes", "design_nodes"),
            "inlet": ("inlet", "inlet_nodes"),
            "outlet": ("outlet", "outlet_nodes"),
        }.get(role, (role, f"{role}_nodes"))
        boundary_entities = None
        for name in entity_names:
            candidate = getattr(self.mesh, name, None)
            if candidate is None:
                continue
            candidate_array = bm.asarray(candidate, dtype=int)
            if candidate_array.size == 0:
                continue
            boundary_entities = candidate_array
            break
        if boundary_entities is None:
            return bm.zeros(space.number_of_global_dofs(), dtype=bool)
        boundary_edges = _boundary_entities_to_edges(self.mesh, boundary_entities)
        if boundary_edges.size == 0:
            return bm.zeros(space.number_of_global_dofs(), dtype=bool)
        edge2dof = bm.asarray(space.edge_to_dof(), dtype=int)
        boundary_dofs = bm.unique(edge2dof[bm.asarray(boundary_edges, dtype=int)].reshape(-1))
        mask = bm.zeros(space.number_of_global_dofs(), dtype=bool)
        mask[boundary_dofs] = True
        return mask

    @cartesian
    def is_inlet_boundary(self, p: Any = None) -> Any:
        if p is None:
            return 0
        if hasattr(p, "edge_to_dof") or hasattr(p, "number_of_global_dofs"):
            return self._boundary_dofs(p, "inlet")
        coords = bm.asarray(p, dtype=float)
        tol = max(1.0e-4, 1.0e-6 * max(abs(float(self.inlet_x)), 
                                       abs(float(self.inlet_ymin)), abs(float(self.inlet_ymax)), 1.0))
        return bm.abs(coords[..., 0] - float(self.inlet_x)) <= tol

    @cartesian
    def is_outlet_boundary(self, p: Any = None) -> Any:
        if p is None:
            return 0
        if hasattr(p, "edge_to_dof") or hasattr(p, "number_of_global_dofs"):
            return self._boundary_dofs(p, "outlet")
        coords = bm.asarray(p, dtype=float)
        tol = max(1.0e-4, 1.0e-6 * max(abs(float(self.inlet_x)), abs(float(self.inlet_ymin)),
                                       abs(float(self.inlet_ymax)), abs(float(self.outlet_x)), 1.0))
        return bm.abs(coords[..., 0] - float(self.outlet_x)) <= tol

    @cartesian
    def is_wall_boundary(self, p: Any = None) -> Any:
        if p is None:
            return 0
        if hasattr(p, "edge_to_dof") or hasattr(p, "number_of_global_dofs"):
            return self._boundary_dofs(p, "wall")
        coords = bm.asarray(p, dtype=float)
        return ~(self.is_inlet_boundary(coords) | self.is_outlet_boundary(coords))

    @cartesian
    def is_velocity_boundary(self, p: Any = None) -> Any:
        if p is None:
            return None
        if hasattr(p, "edge_to_dof") or hasattr(p, "number_of_global_dofs"):
            return self.is_inlet_boundary(p) | self.is_wall_boundary(p)
        coords = bm.asarray(p, dtype=float)
        return ~self.is_outlet_boundary(coords)

    @cartesian
    def is_pressure_boundary(self, p: Any = None) -> Any:
        if p is None:
            return 0 if bool(self.pressure_neumann) else 1
        if hasattr(p, "edge_to_dof") or hasattr(p, "number_of_global_dofs"):
            if bool(self.pressure_neumann):
                return bm.zeros(p.number_of_global_dofs(), dtype=bool)
            return self.is_outlet_boundary(p)
        coords = bm.asarray(p, dtype=float)
        if bool(self.pressure_neumann):
            return bm.zeros(coords[..., 0].shape, dtype=bool)
        return self.is_outlet_boundary(coords)

    @cartesian
    def inlet_velocity(self, p: Any) -> Any:
        return _build_elbow_inflow_profile(
            self.inlet_x,
            self.inlet_ymin,
            self.inlet_ymax,
            self.inlet_max_velocity,
        )(p)

    @cartesian
    def wall_velocity(self, p: Any) -> Any:
        coords = bm.asarray(p, dtype=float)
        return bm.zeros_like(coords, dtype=float)

    @cartesian
    def outlet_pressure(self, p: Any) -> Any:
        coords = bm.asarray(p, dtype=float)
        return bm.zeros(coords[..., 0].shape, dtype=float)

    @cartesian
    def dirichlet_pressure(self, p: Any) -> Any:
        return self.outlet_pressure(p)

    def pressure_integral_target(self) -> float:
        return float(self.pressure_integral_target_value)

    @cartesian
    def dirichlet_velocity(self, p: Any) -> Any:
        coords = bm.asarray(p, dtype=float)
        result = bm.zeros_like(coords, dtype=float)
        inlet = self.inlet_velocity(coords)
        wall = self.wall_velocity(coords)
        inlet_mask = self.is_inlet_boundary(coords)
        wall_mask = self.is_wall_boundary(coords)
        result[inlet_mask] = inlet[inlet_mask]
        result[wall_mask] = wall[wall_mask]
        return result

    @cartesian
    def source(self, p: Any) -> Any:
        coords = bm.asarray(p, dtype=float)
        return bm.zeros_like(coords, dtype=float)


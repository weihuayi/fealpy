"""Shared thin runner utilities for cashocs-style benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Callable, Mapping

from fealpy.backend import backend_manager as bm

from . import adjoint_solver
from .geometry_contract import build_geometry_contract, initialize_optimization_cache
from .benchmark_common import (
    _build_visualization_mesh,
    _build_vtu_point_data,
    _compute_convergence_rates,
    _map_scalar_product,
    get_value,
)
from .geometry_gradient import assemble_geometry_gradient
from .geometry_regularization import _polygon_vertex_normals,_polygon_area_and_centroid
from .mesh_propagation import RemeshResult, propagate_mesh, remesh_mesh, check_mesh_quality
from .objective import evaluate_objective
from .shape_optimizer import OptimizationResult, ShapeOptimizer
from .state_solver import solve_state_system


def _null_shape_derivative_source(*args: Any, **kwargs: Any) -> None:
    return None


@dataclass(slots=True)
class _BenchmarkStateSolver:
    benchmark: Any
    kind: str = "stokes"
    mesh: Any = None
    options: dict[str, Any] = field(default_factory=dict)
    fem: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.benchmark.fluid_model, name)

    def solve_state_system(
        self,
        mesh: Any,
        geometry_contract: Any,
        initial_guess: Any = None,
        options: Any = None,
    ) -> Any:
        self.mesh = mesh
        merged_options = dict(self.options)
        if isinstance(options, Mapping):
            merged_options.update(dict(options))
        self.options = merged_options
        return solve_state_system(
            mesh,
            self,
            geometry_contract,
            initial_guess=initial_guess,
            options=self.options,
        )

    def _build_stationary_stokes_model(self, mesh: Any, options: Mapping[str, Any] | None = None):
        from fealpy.cfd.stationary_incompressible_stokes_lfem_model import (
            StationaryIncompressibleStokesLFEMModel,
        )

        self.mesh = mesh
        resolved_options = dict(options) if isinstance(options, Mapping) else {}
        return StationaryIncompressibleStokesLFEMModel(
            pde=self.benchmark.fluid_model,
            mesh=mesh,
            options=resolved_options or None,
        )

    def _build_stationary_ns_model(self, mesh: Any, options: Mapping[str, Any] | None = None):
        from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import (
            StationaryIncompressibleNSLFEMModel,
        )

        self.mesh = mesh
        resolved_options = dict(options) if isinstance(options, Mapping) else {}
        return StationaryIncompressibleNSLFEMModel(
            pde=self.benchmark.fluid_model,
            mesh=mesh,
            options=resolved_options or None,
        )

    def run(self) -> Any:
        if self.mesh is None:
            raise ValueError("state solver requires a mesh before run()")
        if self.kind == "ns":
            model = self._build_stationary_ns_model(self.mesh, options=self.options)
        else:
            model = self._build_stationary_stokes_model(self.mesh, options=self.options)
        result = model.run()
        self.fem = getattr(model, "fem", None)
        return result


class BenchmarkRunnerMixin:
    """Thin shared runner for benchmark entry scripts."""

    _state_solver_cache: Any = None
    _shape_derivative_builder_cache: Callable[..., Any] | None = None
    _shape_derivative_source_cache: Callable[..., Any] | None = None
    _remesh_handler_cache: Callable[..., Any] | None = None

    def build_state_solver(self) -> Any:
        return self

    def _mesh_boundary_roles(self, mesh: Any) -> dict[str, Any]:
        roles = getattr(mesh, "boundary_nodes_by_role", None) or {}
        return {name: bm.asarray(values, dtype=int).reshape(-1) for name, values in roles.items()}

    def build_adjoint_solver(self):
        return adjoint_solver

    def build_shape_derivative_source(self) -> Callable[..., Any]:
        if self._shape_derivative_source_cache is None:
            self._shape_derivative_source_cache = _null_shape_derivative_source
        return self._shape_derivative_source_cache

    def build_shape_derivative_builder(self) -> Callable[..., Any]:
        return self.build_shape_derivative_source()

    def build_remesh_handler(self) -> Callable[..., Any]:
        if self._remesh_handler_cache is not None:
            return self._remesh_handler_cache
        if not hasattr(self, "mesher") or self.mesher is None:
            raise AttributeError("benchmark runner requires a mesher to rebuild remeshed meshes")

        def remesh_handler(current_state: Any, geometry_contract: Any, propagation_parameters: Any, cache: Any = None) -> Any:
            trial_mesh = get_value(current_state, "trial_mesh", "mesh")
            if trial_mesh is None:
                return False
            design_order = get_value(current_state, "design_boundary_node_order", "design_boundary_node_ids", default=getattr(self, "design_boundary_node_order", ()))
            design_order = tuple(int(node) for node in bm.asarray(design_order, dtype=int).reshape(-1).tolist())
            if len(design_order) == 0:
                return False
            boundary_points = trial_mesh.node[bm.asarray(design_order, dtype=int)]
            remeshed_mesh = self.mesher.build_mesh_from_boundary_points(boundary_points)
            boundary_nodes_by_role = self._mesh_boundary_roles(remeshed_mesh)
            updated_objective_parameters = dict(get_value(current_state, "objective_parameters", default=getattr(self, "objective_parameters", {})))
            updated_objective_parameters["design_boundary_node_order"] = tuple(int(node) for node in bm.asarray(remeshed_mesh.design_boundary_node_order, dtype=int).reshape(-1).tolist())
            updated_objective_parameters["design_boundary_node_ids"] = tuple(int(node) for node in bm.asarray(remeshed_mesh.design_boundary_node_ids, dtype=int).reshape(-1).tolist())
            remesh_objective = get_value(current_state, "trial_objective", "objective", "current_objective")
            return RemeshResult(
                remeshed_state={
                    "mesh": remeshed_mesh,
                    "trial_mesh": remeshed_mesh,
                    "objective_parameters": updated_objective_parameters,
                    "boundary_nodes_by_role": boundary_nodes_by_role,
                    "design_boundary_node_order": tuple(int(node) for node in remeshed_mesh.design_boundary_node_order),
                    "design_boundary_node_ids": tuple(int(node) for node in remeshed_mesh.design_boundary_node_ids),
                    "design_center": tuple(float(value) for value in bm.asarray(getattr(remeshed_mesh, "design_center", getattr(self, "design_center", (0.0, 0.0))), dtype=float).reshape(-1)[:2]),
                    "design_radius": float(getattr(remeshed_mesh, "design_radius", getattr(self, "design_radius", 0.0))),
                    "objective": remesh_objective,
                    "current_objective": remesh_objective,
                    "trial_objective": remesh_objective,
                    "quality_info": get_value(current_state, "quality_info", default=None),
                },
                remeshed=True,
                requires_remesh=False,
                restart_optimization=True,
                quality_info=get_value(current_state, "quality_info", default=None),
                handler_name="build_remesh_handler",
                remesh_output=remeshed_mesh,
                reason="polygon_remesh",
            )

        self._remesh_handler_cache = remesh_handler
        return remesh_handler

    def build_initial_state(self) -> dict[str, Any]:
        state_result = self.build_state_solver().solve_state_system(self.mesh, self.geometry_contract, options=self.options)
        objective_parameters = dict(self.objective_parameters)
        objective_result = evaluate_objective(
            self.mesh,
            state_result,
            objective_parameters,
            current_state={"mesh": self.mesh, "objective_parameters": objective_parameters},
        )
        current_objective = objective_result.total_objective
        if current_objective is not None and hasattr(self, "cache") and self.cache is not None:
            self.cache.reference_objective = float(current_objective)
        return {
            "mesh": self.mesh,
            "fluid_model": self.fluid_model,
            "objective_parameters": objective_parameters,
            "quality_info": None,
            "boundary_nodes_by_role": {key: bm.asarray(value, dtype=int) for key, value in self.boundary_nodes_by_role.items()},
            "design_boundary_node_order": tuple(int(node) for node in self.design_boundary_node_order),
            "design_boundary_node_ids": tuple(int(node) for node in self.design_boundary_node_order),
            "design_center": tuple(float(value) for value in getattr(self, "design_center", (0.0, 0.0))),
            "design_radius": float(getattr(self, "design_radius", 0.0)),
            "state_result": state_result,
            "objective_result": objective_result,
            "objective": current_objective,
            "current_objective": current_objective,
            "iteration": 0,
        }

    def build_mesh_propagator(self):
        if not hasattr(self, "_mesh_propagator_cache") or self._mesh_propagator_cache is None:
            class _MeshPropagator:
                def build_trial_mesh(self, mesh, boundary_displacement, geometry_contract, propagation_parameters, cache=None):
                    return propagate_mesh(mesh, boundary_displacement, geometry_contract, propagation_parameters, cache=cache)

                def check_mesh_quality(self, *args, **kwargs):
                    return check_mesh_quality(*args, **kwargs)

                def remesh_mesh(self, *args, **kwargs):
                    return remesh_mesh(*args, **kwargs)

            self._mesh_propagator_cache = _MeshPropagator()
        return self._mesh_propagator_cache

    def export_vtu_frame(self, path: Path | str, state: Any, step_result: Any = None) -> None:
        state_mesh = state.get("mesh") if isinstance(state, Mapping) else getattr(state, "mesh", None)
        state_result = state.get("state_result") if isinstance(state, Mapping) else getattr(state, "state_result", None)
        if state_mesh is None or state_result is None:
            if state_mesh is None:
                raise ValueError("state must expose mesh for VTU export")
            initial_guess = state.get("initial_guess") if isinstance(state, Mapping) else getattr(state, "initial_guess", None)
            state_result = self.build_state_solver().solve_state_system(state_mesh, self.geometry_contract, initial_guess=initial_guess, options=self.options)
        vis_mesh = _build_visualization_mesh(state_mesh)
        point_data = _build_vtu_point_data(state_mesh, state_result, vis_mesh=vis_mesh, step_result=step_result)
        vis_mesh.nodedata["u"] = point_data["u"]
        vis_mesh.nodedata["p"] = point_data["p"]
        if "shape_d" in point_data:
            vis_mesh.nodedata["shape_d"] = point_data["shape_d"]
        vis_mesh.to_vtk(fname=str(Path(path)))

    def _build_vtu_export_callback(self, output_dir: Path, prefix: str) -> Callable[..., None]:
        def export_callback(*, iteration: int, current_state: Any, step_result: Any, **_: Any) -> None:
            self.export_vtu_frame(output_dir / f"{prefix}_{int(iteration):03d}.vtu", current_state, step_result=step_result)

        return export_callback

    def run(
        self,
        max_iterations: int | None = None,
        *,
        export_vtu: bool = False,
        vtu_output_dir: Path | str | None = None,
        vtu_prefix: str = "iteration",
        export_initial_state: bool = True,
    ) -> OptimizationResult:
        options = dict(self.options)
        if max_iterations is not None:
            options["max_iterations"] = int(max_iterations)
        initial_state = self.build_initial_state()
        if export_vtu:
            output_dir = Path(vtu_output_dir) if vtu_output_dir is not None else Path.cwd() / f"_tmp_shapeopt_frames_{id(self):x}"
            output_dir.mkdir(parents=True, exist_ok=True)
            if export_initial_state:
                self.export_vtu_frame(output_dir / f"{vtu_prefix}_000.vtu", initial_state)
            options["intermediate_result_callback"] = self._build_vtu_export_callback(output_dir, vtu_prefix)
            
        merged_options = dict(self.options)
        if options is not None:
            merged_options.update(dict(options))
        merged_options.setdefault("objective_parameters", self.objective_parameters)
    
        so = ShapeOptimizer(
            geometry_contract=self.geometry_contract,
            state_solver=self.build_state_solver(),
            objective_evaluator=evaluate_objective,
            adjoint_solver=self.build_adjoint_solver(),
            geometry_gradient_assembler=assemble_geometry_gradient,
            mesh_propagator=self.build_mesh_propagator(),
            cache=self.cache,
            options=merged_options,
        )
        return so.run(initial_state)
    
    
    def shape_gradient_test(self, *, h: Any = None, rng: Any = None, verbose: bool = True) -> float:
        custom_rng = rng or random.Random()
        initial_state = self.build_initial_state()
        mesh = initial_state["mesh"]
        objective_parameters = dict(self.objective_parameters)
        objective_parameters["shape_derivative_builder"] = self.build_shape_derivative_builder()
        state_result = initial_state["state_result"]
        objective_result = evaluate_objective(mesh, state_result, objective_parameters, current_state=initial_state)
        shape_gradient = self.build_shape_derivative_builder()(mesh, state_result, objective_parameters, current_state=initial_state)
        design_ids = bm.asarray(self.design_boundary_node_order, dtype=int)
        shape_gradient = {
            int(node): tuple(float(component) for component in bm.asarray(vector, dtype=float).reshape(-1)[:2])
            for node, vector in zip(design_ids.tolist(), bm.asarray(shape_gradient, dtype=float), strict=True)
        }
        if h is None:
            h = {}
            design_coords = mesh.node[design_ids]
            normals = _polygon_vertex_normals(design_coords)
            random_weights = [float(custom_rng.uniform(-1.0, 1.0)) for _ in range(int(design_ids.size))]
            for node_id, normal, weight in zip(design_ids.tolist(), normals, random_weights, strict=True):
                h[int(node_id)] = tuple(float(weight * component) for component in normal)
        current_cost = float(objective_result.total_objective)
        coords = mesh.node
        if coords is None:
            raise ValueError("benchmark mesh does not expose coordinates for Taylor testing")
        coords_arr = bm.asarray(coords, dtype=float)
        length = float(bm.max(coords_arr) - bm.min(coords_arr))
        reference_step = max(1.0e-8, length * 1.0e-6)

        def _evaluate_cost(boundary_displacement: Mapping[int, Any]) -> float:
            trial_state = propagate_mesh(mesh, boundary_displacement, self.geometry_contract, self.cache.propagation_parameters, cache=self.cache)
            if not bool(getattr(trial_state, "accepted", True)):
                raise ValueError("reference trial displacement was rejected")
            trial_mesh = trial_state.trial_mesh if hasattr(trial_state, "trial_mesh") else trial_state
            perturbed_state_result = self.build_state_solver().solve_state_system(trial_mesh, self.geometry_contract, options=self.options)
            perturbed_objective_result = evaluate_objective(
                trial_mesh,
                perturbed_state_result,
                objective_parameters,
                current_state={"mesh": trial_mesh, "objective_parameters": objective_parameters},
            )
            return float(perturbed_objective_result.total_objective)

        try:
            reference_plus = _evaluate_cost({int(node): tuple(float(reference_step * component) for component in bm.asarray(vec, dtype=float).reshape(-1)) for node, vec in h.items()})
            reference_minus = _evaluate_cost({int(node): tuple(float(-reference_step * component) for component in bm.asarray(vec, dtype=float).reshape(-1)) for node, vec in h.items()})
            shape_derivative_h = (reference_plus - reference_minus) / (2.0 * reference_step)
        except Exception:
            shape_derivative_h = _map_scalar_product(shape_gradient, h)

        epsilons = [length * 1.0e-4 / (2**i) for i in range(4)]
        residuals: list[float] = []
        accepted_epsilons: list[float] = []
        for eps in epsilons:
            boundary_displacement = {int(node): tuple(float(eps * component) for component in bm.asarray(vec, dtype=float).reshape(-1)) for node, vec in h.items()}
            trial_state = propagate_mesh(mesh, boundary_displacement, self.geometry_contract, self.cache.propagation_parameters, cache=self.cache)
            if not bool(getattr(trial_state, "accepted", True)):
                continue
            trial_mesh = trial_state.trial_mesh if hasattr(trial_state, "trial_mesh") else trial_state
            perturbed_state_result = self.build_state_solver().solve_state_system(trial_mesh, self.geometry_contract, options=self.options)
            perturbed_objective_result = evaluate_objective(
                trial_mesh,
                perturbed_state_result,
                objective_parameters,
                current_state={"mesh": trial_mesh, "objective_parameters": objective_parameters},
            )
            residuals.append(abs(float(perturbed_objective_result.total_objective) - current_cost - eps * shape_derivative_h))
            accepted_epsilons.append(eps)
        if len(residuals) < 2:
            return float("nan")
        rates = _compute_convergence_rates(accepted_epsilons, residuals)
        if verbose:
            print(f"Taylor test convergence rate: {rates}", flush=True)
        return float(rates[-1])


class ShapeOptimizationRunner(BenchmarkRunnerMixin):
    """Thin orchestration class for a single benchmark case."""

    def __init__(
        self,
        *,
        mesh: Any,
        boundary_info: Mapping[str, Any],
        pde: Any,
        state_solver: Any,
        options: Mapping[str, Any] | None = None,
        objective_parameters: Mapping[str, Any] | None = None,
        cache: Any = None,
        mesher: Any = None,
        geometry_contract: Any = None,
    ) -> None:
        self.mesh = mesh
        self.boundary_info = dict(boundary_info)
        self.boundary_markers = dict(self.boundary_info.get("boundary_markers", {}))
        self.boundary_nodes_by_role = {
            str(name): bm.asarray(values, dtype=int).reshape(-1)
            for name, values in dict(self.boundary_info.get("boundary_nodes_by_role", {})).items()
        }
        self.design_boundary_node_order = tuple(
            int(node)
            for node in bm.asarray(
                self.boundary_info.get("design_boundary_node_order", self.boundary_info.get("design_boundary_node_ids", ())),
                dtype=int,
            ).reshape(-1).tolist()
        )
        self.design_center = tuple(
            float(value)
            for value in bm.asarray(self.boundary_info.get("design_center", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
        )
        self.design_radius = float(self.boundary_info.get("design_radius", 0.0))
        self.fluid_model = pde
        self.state_solver = state_solver
        self.options = dict(options or {})
        self.objective_parameters = dict(objective_parameters or {})
        self.mesher = mesher
        self.geometry_contract = geometry_contract or build_geometry_contract(
            mesh,
            self.boundary_markers,
            spatial_dim=int(self.boundary_info.get("spatial_dim", 2)),
        )
        propagation_parameters = dict(self.boundary_info.get("propagation_parameters", {}))
        if cache is not None:
            self.cache = cache
        elif propagation_parameters:
            self.cache = initialize_optimization_cache(
                reference_mesh=mesh,
                geometry_contract=self.geometry_contract,
                propagation_parameters=propagation_parameters,
                reference_objective=self.boundary_info.get("reference_objective"),
            )
        else:
            self.cache = None
        self._mesh_propagator_cache = None

    def solve_state_system(
        self,
        mesh: Any,
        geometry_contract: Any,
        initial_guess: Any = None,
        options: Any = None,
    ) -> Any:
        merged_options = dict(self.options)
        if isinstance(options, Mapping):
            merged_options.update(dict(options))
        return solve_state_system(
            mesh,
            self.state_solver if self.state_solver is not None else self.fluid_model,
            geometry_contract,
            initial_guess=initial_guess,
            options=merged_options,
        )

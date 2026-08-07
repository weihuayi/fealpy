"""Shape optimizer entry-point scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Number
from typing import Any, Mapping

from fealpy.backend import backend_manager as bm

from .benchmark_common import get_value
from .geometry_contract import refresh_propagation_cache
from .geometry_gradient import build_boundary_displacement
from .l_bfgs import ShapeLBFGSState
from .geometry_contract import refresh_propagation_cache, build_geometry_contract

@dataclass(slots=True)
class OptimizationHistoryEntry:
    """Single optimization history entry."""

    iteration: int
    current_objective: float | None = None
    trial_objective: float | None = None
    step_size: float | None = None
    accepted: bool | None = None
    remeshed: bool = False
    quality_info: Any = None


@dataclass(slots=True)
class OptimizationResult:
    """Full optimization result."""

    final_state: Any
    final_cache: Any
    history: list[OptimizationHistoryEntry] = field(default_factory=list)
    initial_state: Any = None
    initial_cache: Any = None
    iterations_run: int = 0
    terminated_early: bool = False
    terminated_reason: str | None = None
    stop_iteration: int | None = None
    last_step_result: Any = None

    @property
    def final_objective(self) -> float | None:
        state = self.final_state
        if isinstance(state, Mapping):
            objective = state.get("objective", state.get("current_objective"))
        else:
            objective = getattr(state, "objective", None)
            if objective is None:
                objective = getattr(state, "current_objective", None)
        if objective is not None:
            return float(objective)
        return next(
            (
                float(entry.trial_objective)
                if entry.accepted and entry.trial_objective is not None
                else float(entry.current_objective)
                for entry in reversed(self.history)
                if (entry.accepted and entry.trial_objective is not None) or entry.current_objective is not None
            ),
            None,
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self) -> list[tuple[str, Any]]:
        return [
            ("initial_state", self.initial_state),
            ("initial_cache", self.initial_cache),
            ("final_state", self.final_state),
            ("final_objective", self.final_objective),
            ("final_cache", self.final_cache),
            ("history", self.history),
            ("iterations_run", self.iterations_run),
            ("terminated_early", self.terminated_early),
            ("terminated_reason", self.terminated_reason),
            ("stop_iteration", self.stop_iteration),
            ("last_step_result", self.last_step_result),
        ]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.items())


@dataclass(slots=True)
class OneStepOptimizationCheckResult:
    """Finite-difference check for one optimization step."""

    current_objective: float
    trial_objective: float
    finite_difference: float
    objective_decrease: float
    accepted: bool


@dataclass(slots=True)
class LineSearchResult:
    """Backtracking line-search result."""

    initial_step_size: float
    step_size: float
    current_objective: float | None
    trial_objective: float | None
    accepted: bool
    requires_remesh: bool
    boundary_displacement: Any
    trial_state: Any
    quality_info: Any = None
    trial_state_result: Any = None
    trial_objective_result: Any = None


@dataclass(slots=True)
class RemeshResult:
    """Remeshing audit result."""

    remeshed_state: Any
    remeshed: bool
    requires_remesh: bool
    restart_optimization: bool = False
    quality_info: Any = None
    handler_name: str | None = None
    remesh_output: Any = None
    reason: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self) -> list[tuple[str, Any]]:
        return [
            ("remeshed_state", self.remeshed_state),
            ("remeshed", self.remeshed),
            ("requires_remesh", self.requires_remesh),
            ("restart_optimization", self.restart_optimization),
            ("quality_info", self.quality_info),
            ("handler_name", self.handler_name),
            ("remesh_output", self.remesh_output),
            ("reason", self.reason),
        ]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.items())


@dataclass(slots=True)
class StepResult:
    """One optimization step result."""

    state_result: Any
    objective_result: Any
    adjoint_rhs_source: Any
    adjoint_result: Any
    geometry_gradient_result: Any
    boundary_displacement: Any
    trial_state: Any
    accepted_state: Any
    current_objective: float | None
    trial_objective: float | None
    accepted: bool
    requires_remesh: bool
    step_size: float
    quality_info: Any = None
    line_search_result: LineSearchResult | None = None
    remesh_result: RemeshResult | None = None
    history_entry: OptimizationHistoryEntry | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self) -> list[tuple[str, Any]]:
        return [
            ("state_result", self.state_result),
            ("objective_result", self.objective_result),
            ("adjoint_rhs_source", self.adjoint_rhs_source),
            ("adjoint_result", self.adjoint_result),
            ("geometry_gradient_result", self.geometry_gradient_result),
            ("boundary_displacement", self.boundary_displacement),
            ("trial_state", self.trial_state),
            ("accepted_state", self.accepted_state),
            ("current_objective", self.current_objective),
            ("trial_objective", self.trial_objective),
            ("accepted", self.accepted),
            ("requires_remesh", self.requires_remesh),
            ("step_size", self.step_size),
            ("quality_info", self.quality_info),
            ("line_search_result", self.line_search_result),
            ("remesh_result", self.remesh_result),
            ("history_entry", self.history_entry),
        ]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.items())


class ShapeOptimizer:
    """Shape optimization driver."""

    get_value = staticmethod(get_value)

    def __init__(
        self,
        geometry_contract: Any,
        state_solver: Any,
        objective_evaluator: Any,
        adjoint_solver: Any,
        geometry_gradient_assembler: Any,
        mesh_propagator: Any,
        cache: Any = None,
        options: Any = None,
    ) -> None:
        self.geometry_contract = geometry_contract
        self.state_solver = state_solver
        self.objective_evaluator = objective_evaluator
        self.adjoint_solver = adjoint_solver
        self.geometry_gradient_assembler = geometry_gradient_assembler
        self.mesh_propagator = mesh_propagator
        self.cache = cache
        self.options = options
        self._initial_step_size = self._resolve_float_option("initial_step_size", default=1.0)
        self._armijo_epsilon_value = self._resolve_float_option("epsilon_armijo", default=1.0e-4)
        self._line_search_reduction_value = self._resolve_float_option("line_search_reduction", "beta_armijo", default=0.5)
        self._minimum_step_size_value = self._resolve_float_option("minimum_step_size", "min_step_size", default=1.0e-12)
        self._max_line_search_iterations_value = self._resolve_int_option(
            "max_line_search_iterations",
            "line_search_max_iterations",
            default=20,
        )
        self._remesh_iteration_interval_value = self._resolve_int_option("remesh_iter", default=0)
        self._lbfgs_memory_size_value = self._resolve_int_option(
            "lbfgs_memory_size","memory_size","NumGrad",default=10,minimum=1,
        )
        self._objective_parameters_default = self.get_value(self.options, "objective_parameters", default=None)
        self.algorithm = str(self.get_value(self.options, "algorithm", default="lbfgs")).casefold()
        self.lbfgs_state = ShapeLBFGSState(memory_size=self._lbfgs_memory_size_value) if self.algorithm == "lbfgs" else None
        self._lbfgs_pending_update: tuple[Any, Any] | None = None
        self._component_method_cache: dict[tuple[int, str], Any] = {}

    def _resolve_float_option(self, *names: str, default: float) -> float:
        value = self.get_value(self.options, *names, default=default)
        return float(default if value is None else value)

    def _resolve_int_option(self, *names: str, default: int, minimum: int | None = None) -> int:
        value = self.get_value(self.options, *names, default=default)
        resolved = int(default if value is None else value)
        if minimum is not None and resolved < minimum:
            return int(minimum)
        return resolved

    def _invoke_component(self, component: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a component method with caching."""
        method = self._resolve_component_method(component, method_name)
        return method(*args, **kwargs)

    def _resolve_component_method(self, component: Any, method_name: str) -> Any:
        """Resolve a component method or callable with caching."""
        if component is None:
            raise NotImplementedError(f"{method_name} ??????????????????????")
        cache_key = (id(component), method_name)
        method = self._component_method_cache.get(cache_key)
        if method is not None:
            return method
        method = getattr(component, method_name, None)
        if callable(method):
            self._component_method_cache[cache_key] = method
            return method
        if callable(component):
            self._component_method_cache[cache_key] = component
            return component
        raise NotImplementedError(f"{method_name} ??????????????????????????????")

    def _resolve_state_context(self, current_state: Any) -> tuple[Any, Any, Any]:
        """Resolve mesh, objective parameters, and current objective in one pass."""
        mesh = self.get_value(current_state, "mesh", "trial_mesh")
        if mesh is None:
            mesh = current_state
        objective_parameters = self.get_value(current_state, "objective_parameters")
        if objective_parameters is None:
            objective_parameters = self._objective_parameters_default if self._objective_parameters_default is not None else {}
        current_objective = self.get_value(current_state, "objective", "current_objective")
        if current_objective is None and self.cache is not None:
            current_objective = self.get_value(self.cache, "reference_objective")
        if current_objective is not None:
            current_objective = float(current_objective)
        return mesh, objective_parameters, current_objective

    def _resolve_current_state_result(self, current_state: Any, mesh: Any) -> Any:
        """Reuse a cached state result when it is still valid for the current mesh."""
        cached_state_result = self.get_value(current_state, "state_result", default=None)
        if cached_state_result is None:
            return None
        if not bool(self.get_value(current_state, "state_solution_valid", default=False)):
            return None
        cached_mesh = self.get_value(cached_state_result, "mesh", default=None)
        if cached_mesh is not None and cached_mesh != mesh:
            return None
        return cached_state_result

    def _negate_value(self, value: Any) -> Any:
        """Negate a scalar, vector, or nested mapping."""
        if isinstance(value, Number):
            return -value
        if isinstance(value, Mapping):
            return {key: self._negate_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._negate_value(item) for item in value)
        if isinstance(value, list):
            return [self._negate_value(item) for item in value]
        return -value

    def _apply_pending_lbfgs_update(self, current_gradient: Any) -> None:
        """Apply the pending LBFGS update if one exists."""
        if self.lbfgs_state is None or self._lbfgs_pending_update is None:
            return
        step, previous_gradient = self._lbfgs_pending_update
        self.lbfgs_state.update_history(step, previous_gradient, current_gradient)
        self._lbfgs_pending_update = None

    def _store_pending_lbfgs_update(self, step: Any, previous_gradient: Any) -> None:
        """Store the step for the next LBFGS update."""
        if self.lbfgs_state is None:
            return
        self._lbfgs_pending_update = (step, previous_gradient)

    def _scalar_measure(self, value: Any) -> float:
        """Scale a scalar, vector, or mapping."""
        if value is None:
            return 0.0
        if isinstance(value, Number):
            return float(value)
        if isinstance(value, Mapping):
            return float(sum(map(self._scalar_measure, value.values())))
        if isinstance(value, (tuple, list)):
            return float(sum(map(self._scalar_measure, value)))
        array = bm.asarray(value, dtype=float)
        return float(bm.sum(array))

    def _dot_measure(self, left: Any, right: Any) -> float:
        """Compute a dot-like measure between two gradient representations."""
        if left is None or right is None:
            return 0.0
        if isinstance(left, Number) and isinstance(right, Number):
            return float(left) * float(right)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return float(sum(self._dot_measure(value, right.get(key, 0.0)) for key, value in left.items()))
        if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
            return float(sum(self._dot_measure(left_item, right_item) for left_item, right_item in zip(left, right)))
        if isinstance(left, Mapping) and isinstance(right, Number):
            return float(right) * float(sum(map(self._scalar_measure, left.values())))
        if isinstance(right, Mapping) and isinstance(left, Number):
            return float(left) * float(sum(map(self._scalar_measure, right.values())))
        if isinstance(left, (tuple, list)) and isinstance(right, Number):
            return float(right) * float(sum(map(self._scalar_measure, left)))
        if isinstance(right, (tuple, list)) and isinstance(left, Number):
            return float(left) * float(sum(map(self._scalar_measure, right)))
        return self._scalar_measure(left) * self._scalar_measure(right)

    def _directional_decrease_measure(self, geometry_gradient_result: Any) -> float:
        """Extract the directional derivative used by Armijo."""
        directional_derivative = self.get_value(
            geometry_gradient_result,
            "directional_derivative",
            "decrease_measure",
            default=None,
        )
        if directional_derivative is not None:
            return self._scalar_measure(directional_derivative)
        armijo_gradient = self.get_value(
            geometry_gradient_result,
            "propagated_gradient",
            "node_gradient",
            "normal_gradient",
            "raw_gradient",
            default=None,
        )
        descent_direction = self.get_value(geometry_gradient_result, "descent_direction", default=None)
        if armijo_gradient is None or descent_direction is None:
            return 0.0
        return self._dot_measure(armijo_gradient, descent_direction)

    def _evaluate_trial_objective_along_direction(
        self,
        mesh: Any,
        current_state: Any,
        descent_direction: Any,
        step_size: float,
    ) -> float | None:
        """Evaluate the actual trial objective for a fixed step size."""
        propagation_parameters = self.get_value(self.cache, "propagation_parameters", default={})
        objective_parameters = self.get_value(current_state, "objective_parameters")
        if objective_parameters is None:
            objective_parameters = self._objective_parameters_default if self._objective_parameters_default is not None else {}

        boundary_displacement = build_boundary_displacement(
            descent_direction,
            step_size,
            self.geometry_contract,
            self.options,
            mesh=mesh,
            cache=self.cache,
            objective_parameters=objective_parameters,
        )
        trial_state = self._invoke_component(
            self.mesh_propagator,
            "build_trial_mesh",
            mesh,
            boundary_displacement,
            self.geometry_contract,
            propagation_parameters,
            cache=self.cache,
        )
        quality_info = self.get_value(trial_state, "quality_info")
        quality_state = self._quality_state(quality_info)
        quality_flag = self.get_value(quality_info, "accepted", default=None)
        if quality_state in {"rejected", "invalid"}:
            return None
        if bool(self.get_value(quality_info, "has_negative_cells", default=False)):
            return None
        if quality_flag is False and quality_state not in {"good", "marginal", "accepted", "poor", None}:
            return None

        trial_mesh = self.get_value(trial_state, "trial_mesh", "mesh")
        if trial_mesh is None:
            return None

        trial_state_result = self._invoke_component(
            self.state_solver,
            "solve_state_system",
            trial_mesh,
            self.geometry_contract,
            initial_guess=self.get_value(current_state, "initial_guess"),
            options=self.options,
        )
        if getattr(trial_state_result, "converged", None) is False:
            return None

        trial_objective_result = self._invoke_component(
            self.objective_evaluator,
            "evaluate_objective",
            trial_mesh,
            trial_state_result,
            objective_parameters,
            current_state={"mesh": trial_mesh, "objective_parameters": objective_parameters},
        )
        trial_objective = self.get_value(trial_objective_result, "total_objective", "objective", default=None)
        return None if trial_objective is None else float(trial_objective)

    def _finite_difference_directional_decrease_measure(
        self,
        mesh: Any,
        current_state: Any,
        descent_direction: Any,
        current_objective: float | None,
    ) -> float | None:
        """Estimate the Armijo directional derivative along the actual trial path."""
        if descent_direction is None:
            return None
        if current_objective is None:
            return None

        step = abs(float(self.get_value(self.options, "armijo_fd_step_size", default=1.0e-6)))
        if step <= 0.0:
            return None

        plus_objective = self._evaluate_trial_objective_along_direction(
            mesh,
            current_state,
            descent_direction,
            step,
        )
        if plus_objective is not None:
            return (float(plus_objective) - float(current_objective)) / step
        return None

    def _quality_state(self, quality_info: Any) -> str | None:
        """Return the normalized mesh quality state."""
        quality_state = self.get_value(quality_info, "quality_state", "status", default=None)
        if quality_state is None:
            return None
        return str(quality_state).casefold()

    def _mark_trial_state_failed(self, quality_info: Any, reason: str) -> dict[str, Any]:
        """Return quality metadata that forces line-search rejection."""
        if isinstance(quality_info, Mapping):
            updated = dict(quality_info)
        else:
            updated = {}
            if quality_info is not None:
                updated["original_quality_info"] = quality_info
        updated["quality_state"] = "invalid"
        updated["status"] = "rejected"
        updated["accepted"] = False
        updated["reason"] = reason
        return updated

    def _annotate_iteration(self, state: Any, iteration: int) -> Any:
        """Attach the iteration number to the state."""
        if isinstance(state, Mapping):
            annotated = dict(state)
            annotated["iteration"] = iteration
            return annotated
        setattr(state, "iteration", iteration)
        return state

    def _requires_remesh(self, current_state: Any, quality_info: Any) -> bool:
        """Decide whether the accepted state needs remeshing."""
        quality_state = self._quality_state(quality_info)
        if quality_state in {"poor", "rejected", "invalid"}:
            return True
        if bool(self.get_value(quality_info, "needs_remesh", default=False)):
            return True
        return False

    def _reset_lbfgs_state(self) -> None:
        """Reset the LBFGS history after a remesh."""
        if self.lbfgs_state is not None:
            self.lbfgs_state.reset()
        self._lbfgs_pending_update = None

    def _remesh_accepted_state(
        self,
        accepted_state: Any,
        quality_info: Any,
        current_state: Any,
    ) -> RemeshResult:
        """Remesh an accepted state when quality requires it."""
        requires_remesh = self._requires_remesh(current_state, quality_info)
        if not requires_remesh:
            return RemeshResult(
                remeshed_state=accepted_state,
                remeshed=False,
                requires_remesh=False,
                quality_info=quality_info,
                reason="quality_ok",
            )

        method_name, method = next(
            (
                (name, getattr(self.mesh_propagator, name, None))
                for name in ("remesh_trial_mesh", "remesh_mesh", "remesh")
                if callable(getattr(self.mesh_propagator, name, None))
            ),
            (None, None),
        )
        if method is not None:
            remesh_output = method(
                accepted_state,
                self.geometry_contract,
                self.get_value(self.cache, "propagation_parameters", default={}),
                cache=self.cache,
            )
            if remesh_output is False:
                method = None
            elif hasattr(remesh_output, "remeshed_state"):
                return remesh_output
            elif remesh_output is True or remesh_output is None:
                return RemeshResult(
                    remeshed_state=accepted_state,
                    remeshed=True,
                    requires_remesh=True,
                    restart_optimization=True,
                    quality_info=quality_info,
                    handler_name=method_name,
                    remesh_output=remesh_output,
                )
            else:
                return RemeshResult(
                    remeshed_state=remesh_output,
                    remeshed=True,
                    requires_remesh=True,
                    restart_optimization=True,
                    quality_info=quality_info,
                    handler_name=method_name,
                    remesh_output=remesh_output,
                )

        updated_state = accepted_state
        if isinstance(updated_state, Mapping):
            updated_state = dict(updated_state)
            updated_state["requires_remesh"] = True
        else:
            setattr(updated_state, "requires_remesh", True)
        return RemeshResult(
            remeshed_state=updated_state,
            remeshed=False,
            requires_remesh=True,
            restart_optimization=False,
            quality_info=quality_info,
            reason="remesh_requested_but_no_hook",
        )

    def _backtracking_trial(
        self,
        mesh: Any,
        current_state: Any,
        descent_direction: Any,
        current_objective: float | None,
        initial_step_size: float | None = None,
        directional_derivative: float | None = None,
    ) -> LineSearchResult:
        """Run one backtracking line-search trial."""
        propagation_parameters = self.get_value(self.cache, "propagation_parameters", default={})
        objective_parameters = self.get_value(current_state, "objective_parameters")
        if objective_parameters is None:
            objective_parameters = self._objective_parameters_default if self._objective_parameters_default is not None else {}
        step_size = self._initial_step_size
        reduction_factor = self._line_search_reduction_value
        min_step_size = self._minimum_step_size_value
        max_iterations = self._max_line_search_iterations_value
        diagnostics = bool(self.get_value(self.options, "diagnostics", "optimization_diagnostics", default=False))

        trial_state = None
        trial_objective = None
        accepted = False
        quality_info = None
        boundary_displacement = None
        line_search_iteration = 0
        backtracked = False
        use_fd_on_reject = bool(
            self.get_value(
                self.options,
                "armijo_use_finite_difference_directional_derivative_on_reject",
                default=False,
            )
        )
        fd_directional_derivative_used = False

        while line_search_iteration < max_iterations and step_size >= min_step_size:
            if diagnostics:
                print(f"[opt-backtrack] trial={line_search_iteration + 1} step={step_size:.6e}")
            boundary_displacement = build_boundary_displacement(
                descent_direction,
                step_size,
                self.geometry_contract,
                self.options,
                mesh=mesh,
                cache=self.cache,
                objective_parameters=objective_parameters,
            )
            trial_state = self._invoke_component(
                self.mesh_propagator,
                "build_trial_mesh",
                mesh,
                boundary_displacement,
                self.geometry_contract,
                propagation_parameters,
                cache=self.cache,
            )
            quality_info = self.get_value(trial_state, "quality_info")
            trial_mesh = self.get_value(trial_state, "trial_mesh", "mesh")
            trial_objective = self.get_value(trial_state, "trial_objective")
            quality_state = self._quality_state(quality_info)
            quality_flag = self.get_value(quality_info, "accepted", default=None)
            definitely_rejected = quality_state in {"rejected", "invalid"}
            if quality_flag is False and quality_state not in {"good", "marginal", "accepted", "poor", None}:
                definitely_rejected = True
            
            if definitely_rejected:
                trial_objective = current_objective
            elif trial_mesh is not None and quality_info is not None:
                trial_objective_result = None
                trial_state_result = self._invoke_component(
                    self.state_solver,
                    "solve_state_system",
                    trial_mesh,
                    self.geometry_contract,
                    initial_guess=self.get_value(current_state, "initial_guess"),
                    options=self.options,
                )
                if getattr(trial_state_result, "converged", None) is False:
                    quality_info = self._mark_trial_state_failed(quality_info, "state_solve_not_converged")
                    trial_objective = current_objective
                    if diagnostics:
                        iterations = getattr(trial_state_result, "nonlinear_iterations", None)
                        residual_u = getattr(trial_state_result, "final_residual_u", None)
                        residual_p = getattr(trial_state_result, "final_residual_p", None)
                        print(
                            "[opt-backtrack] trial state rejected: "
                            f"not converged iterations={iterations} "
                            f"res_u={residual_u} res_p={residual_p}"
                        )
                elif trial_objective is None:
                    trial_objective_result = self._invoke_component(
                        self.objective_evaluator,
                        "evaluate_objective",
                        trial_mesh,
                        trial_state_result,
                        objective_parameters,
                        current_state={"mesh": trial_mesh, "objective_parameters": objective_parameters},
                    )
                    evaluated_trial_objective = self.get_value(trial_objective_result, "total_objective", "objective", default=None)
                    if evaluated_trial_objective is not None:
                        trial_objective = float(evaluated_trial_objective)
            if trial_objective is None:
                trial_objective = current_objective
            accepted = self.accept_trial_update(
                current_objective,
                trial_objective,
                step_size,
                directional_derivative=directional_derivative,
                quality_info=quality_info,
            )
            if (
                not accepted
                and use_fd_on_reject
                and not fd_directional_derivative_used
                and trial_objective is not None
                and current_objective is not None
                and quality_state not in {"rejected", "invalid"}
            ):
                fd_directional_derivative = self._finite_difference_directional_decrease_measure(
                    mesh,
                    current_state,
                    descent_direction,
                    current_objective,
                )
                if fd_directional_derivative is not None:
                    directional_derivative = fd_directional_derivative
                    fd_directional_derivative_used = True
                    accepted = self.accept_trial_update(
                        current_objective,
                        trial_objective,
                        step_size,
                        directional_derivative=directional_derivative,
                        quality_info=quality_info,
                    )
            if diagnostics:
                quality_state = self._quality_state(quality_info)
                print(
                    f"[opt-backtrack] trial={line_search_iteration + 1} accepted={accepted} "
                    f"quality={quality_state} current={current_objective} trial={trial_objective}"
                )
            if accepted:
                break
            step_size *= reduction_factor
            line_search_iteration += 1
            backtracked = True

        if trial_state is None:
            trial_state = {
                "trial_mesh": mesh,
                "trial_objective": current_objective,
                "quality_info": None,
            }

        requires_remesh = bool(accepted and (backtracked or self._requires_remesh(current_state, quality_info)))

        return LineSearchResult(
            initial_step_size=self._initial_step_size,
            step_size=step_size,
            current_objective=current_objective,
            trial_objective=trial_objective,
            accepted=accepted,
            requires_remesh=requires_remesh,
            boundary_displacement=boundary_displacement,
            trial_state=trial_state,
            quality_info=quality_info,
            trial_state_result=trial_state_result,
            trial_objective_result=trial_objective_result,
        )

    def run(self, initial_state: Any) -> OptimizationResult:
        """Run the full optimization loop."""
        max_iterations = int(self.get_value(self.options, "max_iterations", default=1))
        intermediate_result_callback = self.get_value(self.options, "intermediate_result_callback")
        current_state = initial_state
        initial_cache = self.cache
        history: list[OptimizationHistoryEntry] = []
        last_step_result: StepResult | None = None
        terminated_early = False
        terminated_reason: str | None = None
        for iteration in range(max_iterations):
            iteration_state = self._annotate_iteration(current_state, iteration + 1)
            step_result = self.step(iteration_state)
            last_step_result = step_result
            history.append(step_result.history_entry)
            if step_result.accepted:
                current_state = step_result.accepted_state
            else:
                terminated_early = True
                terminated_reason = "trial_rejected"
                break
            if callable(intermediate_result_callback):
                intermediate_result_callback(iteration=iteration + 1, step_result=step_result, current_state=current_state)
        if terminated_reason is None:
            terminated_reason = "max_iterations_zero" if max_iterations <= 0 else "max_iterations_reached"
        return OptimizationResult(
            initial_state=initial_state,
            initial_cache=initial_cache,
            final_state=current_state,
            final_cache=self.cache,
            history=history,
            iterations_run=len(history),
            terminated_early=terminated_early,
            terminated_reason=terminated_reason,
            stop_iteration=0 if max_iterations <= 0 else (len(history) if history else None),
            last_step_result=last_step_result,
        )

    def step(self, current_state: Any) -> StepResult:
        """Run one optimization step."""
        mesh, objective_parameters, current_objective = self._resolve_state_context(current_state)
        state_result = self._resolve_current_state_result(current_state, mesh)
        if state_result is None:
            state_result = self._invoke_component(
                self.state_solver,
                "solve_state_system",
                mesh,
                self.geometry_contract,
                initial_guess=self.get_value(current_state, "initial_guess"),
                options=self.options,
            )
        objective_result = self._invoke_component(
            self.objective_evaluator,
            "evaluate_objective",
            mesh,
            state_result,
            objective_parameters,
            current_state=current_state,
        )
        if current_objective is None:
            current_objective = self.get_value(objective_result, "total_objective", "objective", default=None)
            if current_objective is not None:
                current_objective = float(current_objective)
        adjoint_rhs_source = self.get_value(objective_result, "adjoint_rhs_source", default=None)
        if adjoint_rhs_source is None:
            adjoint_rhs_source = self.get_value(objective_result, "adjoint_rhs", default=None)
        adjoint_result = self._invoke_component(
            self.adjoint_solver,
            "solve_adjoint_system",
            mesh,
            state_result,
            adjoint_rhs_source,
            self.geometry_contract,
            options=self.options,
        )
        geometry_gradient_result = self._invoke_component(
            self.geometry_gradient_assembler,
            "assemble_geometry_gradient",
            mesh,
            state_result,
            adjoint_result,
            self.geometry_contract,
            self.cache,
            objective_result,
            options=self.options,
        )

        lbfgs_gradient = self.get_value(geometry_gradient_result, "descent_direction", default=None)
        if lbfgs_gradient is not None:
            lbfgs_gradient = self._negate_value(lbfgs_gradient)
        else:
            lbfgs_gradient = next((self.get_value(geometry_gradient_result, name, default=None) for name in ("raw_gradient", "propagated_gradient", "normal_gradient", "node_gradient")), None)
        if self.lbfgs_state is not None and lbfgs_gradient is not None:
            self._apply_pending_lbfgs_update(lbfgs_gradient)
            descent_direction = self.lbfgs_state.compute_direction(lbfgs_gradient)
            geometry_gradient_result = replace(geometry_gradient_result, descent_direction=descent_direction)
        else:
            descent_direction = self.get_value(geometry_gradient_result, "descent_direction")
            if descent_direction is None:
                descent_direction = next((self.get_value(geometry_gradient_result, name, default=None) for name in ("node_gradient", "normal_gradient")), None)

        if bool(self.get_value(self.options, "armijo_use_finite_difference_directional_derivative", default=False)):
            decrease_measure = self._finite_difference_directional_decrease_measure(
                mesh,
                current_state,
                descent_direction,
                current_objective,
            )
            if decrease_measure is None:
                decrease_measure = self._directional_decrease_measure(geometry_gradient_result)
        else:
            decrease_measure = self._directional_decrease_measure(geometry_gradient_result)
        if bool(self.get_value(self.options, "strict_normal_boundary_update", default=False)):
            decrease_measure = None
        search_result = self._backtracking_trial(
            mesh,
            current_state,
            descent_direction,
            current_objective,
            initial_step_size=None,
            directional_derivative=decrease_measure,
        )
        
        trial_state = search_result.trial_state
        trial_objective = search_result.trial_objective
        accepted = search_result.accepted
        quality_info = search_result.quality_info
        boundary_displacement = search_result.boundary_displacement
        step_size = search_result.step_size
        remesh_result = RemeshResult(
            remeshed_state=trial_state,
            remeshed=False,
            requires_remesh=False,
            quality_info=quality_info,
            reason="not_accepted",
        )
        
        requires_remesh = False
        if not accepted:
            accepted_state = current_state
        else:
            remesh_result = self._remesh_accepted_state(
                trial_state,
                quality_info,
                current_state,
            )
            requires_remesh = bool(remesh_result.requires_remesh)
            accepted_state_base = remesh_result.remeshed_state
            remeshed = bool(remesh_result.remeshed or remesh_result.restart_optimization)
            if remeshed:
                objective_parameters = self.get_value(
                    accepted_state_base,
                    "objective_parameters",
                    default=objective_parameters,
                )
                new_mesh = self.get_value(accepted_state_base, "mesh", "trial_mesh")
                boundary_roles = self.get_value(
                    accepted_state_base,
                    "boundary_nodes_by_role",
                    default=None,
                )
                if new_mesh is not None and isinstance(boundary_roles, Mapping):
                    self.geometry_contract = build_geometry_contract(
                        new_mesh,
                        boundary_roles,
                        spatial_dim=self.get_value(
                            self.geometry_contract,
                            "spatial_dim",
                            default=2,
                        ),
                    )
                if isinstance(self.options, dict):
                    self.options["objective_parameters"] = objective_parameters
                accepted_state = self.update_current_state(
                    accepted_state_base,
                    accepted_objective=None,
                    objective_parameters=objective_parameters,
                    state_result=None,
                    initial_guess=None,
                )
                self._reset_lbfgs_state()

            else:
                accepted_objective = self.get_value(
                    accepted_state_base,
                    "objective",
                    "current_objective",
                    default=trial_objective,
                )
                trial_state_result = self.get_value(
                    search_result,
                    "trial_state_result",
                    default=None,
                )
                accepted_state = self.update_current_state(
                    accepted_state_base,
                    accepted_objective=accepted_objective,
                    objective_parameters=objective_parameters,
                    state_result=trial_state_result,
                    initial_guess=trial_state_result,
                )
                self._store_pending_lbfgs_update(
                    boundary_displacement,
                    lbfgs_gradient,
                )

        history_entry = self.record_iteration_log(
            iteration=self.get_value(current_state, "iteration", "iteration_index", default=None),
            current_objective=current_objective,
            trial_objective=trial_objective,
            step_size=step_size,
            accepted=accepted,
            remeshed=bool(remesh_result.remeshed),
            quality_info=quality_info,
        )

        return StepResult(
            state_result=state_result,
            objective_result=objective_result,
            adjoint_rhs_source=adjoint_rhs_source,
            adjoint_result=adjoint_result,
            geometry_gradient_result=geometry_gradient_result,
            boundary_displacement=boundary_displacement,
            trial_state=trial_state,
            accepted_state=accepted_state,
            current_objective=current_objective,
            trial_objective=trial_objective,
            accepted=accepted,
            requires_remesh=requires_remesh,
            step_size=step_size,
            quality_info=quality_info,
            line_search_result=search_result,
            remesh_result=remesh_result,
            history_entry=history_entry,
        )

    def line_search(
        self,
        current_state: Any,
        descent_direction: Any,
        step_size: float,
    ) -> Any:
        """Run a line search for a given direction."""
        mesh, _, current_objective = self._resolve_state_context(current_state)
        if current_objective is None:
            mesh, objective_parameters, _ = self._resolve_state_context(current_state)
            state_result = self._resolve_current_state_result(current_state, mesh)
            if state_result is None:
                state_result = self._invoke_component(
                    self.state_solver,
                    "solve_state_system",
                    mesh,
                    self.geometry_contract,
                    initial_guess=self.get_value(current_state, "initial_guess"),
                    options=self.options,
                )
            objective_result = self._invoke_component(
                self.objective_evaluator,
                "evaluate_objective",
                mesh,
                state_result,
                objective_parameters,
                current_state=current_state,
            )
            current_objective = self.get_value(objective_result, "total_objective", "objective", default=None)

        search_result = self._backtracking_trial(
            mesh,
            current_state,
            descent_direction,
            current_objective,
            initial_step_size=step_size,
            directional_derivative=None,
        )
        return search_result

    def accept_trial_update(
        self,
        current_objective: float | None,
        trial_objective: float | None,
        step_size: float,
        directional_derivative: float | None = None,
        quality_info: Any = None,
    ) -> bool:
        """Decide whether to accept a trial update."""
        if trial_objective is None:
            return False

        quality_state = self._quality_state(quality_info)
        if quality_state in {"rejected", "invalid"}:
            return False
        if quality_info is not None:
            quality_reason = self.get_value(quality_info, "reason", default=None)
            has_negative_cells = bool(self.get_value(quality_info, "has_negative_cells", default=False))
            if quality_reason == "negative_cells" or has_negative_cells:
                return False
            accepted_flag = self.get_value(quality_info, "accepted", default=None)
            if accepted_flag is False and quality_state not in {"good", "marginal", "accepted", "poor", None}:
                return False

        if current_objective is None:
            return True

        decrease_measure = 0.0
        if directional_derivative is not None:
            decrease_measure = self._scalar_measure(directional_derivative)
        armijo_rhs = current_objective + self._armijo_epsilon_value * step_size * decrease_measure
        configured_tolerance = max(float(self.get_value(self.options, "objective_tolerance", default=0.0)), 0.0)
        scale = max(1.0, abs(float(current_objective)), abs(float(trial_objective)))
        objective_tolerance = max(configured_tolerance, 1.0e-9 * scale)
        return trial_objective <= armijo_rhs + objective_tolerance
    
    def update_current_state(
        self,
        accepted_state: Any,
        accepted_objective: float | None = None,
        objective_parameters: Any = None,
        state_result: Any = None,
        initial_guess: Any = None,
    ) -> Any:
        """Refresh the current accepted state."""
        accepted_mesh = self.get_value(accepted_state, "mesh", "trial_mesh")
        if accepted_objective is None:
            accepted_objective = self.get_value(accepted_state, "objective", "trial_objective")
        accepted_objective_parameters = self.get_value(accepted_state, "objective_parameters")
        if accepted_objective_parameters is not None:
            objective_parameters = accepted_objective_parameters
        if objective_parameters is None:
            objective_parameters = self.get_value(self.options, "objective_parameters", default=None)
        clear_reference_objective = accepted_mesh is not None and accepted_objective is None
        if self.cache is not None and accepted_mesh is not None:
            self.cache = refresh_propagation_cache(
                self.cache,
                {"mesh": accepted_mesh},
                accepted_objective=accepted_objective,
                invalidate_solution_cache=True,
                clear_reference_objective=clear_reference_objective,
            )
        self._invalidate_component_memory()
        if isinstance(accepted_state, Mapping):
            updated_state = dict(accepted_state)
            if accepted_mesh is not None:
                updated_state["mesh"] = accepted_mesh
            if accepted_objective is not None:
                updated_state["objective"] = accepted_objective
                updated_state["current_objective"] = accepted_objective
            else:
                updated_state.pop("objective", None)
                updated_state.pop("current_objective", None)
            if objective_parameters is not None:
                updated_state["objective_parameters"] = objective_parameters
            elif "objective_parameters" not in updated_state and self.get_value(self.options, "objective_parameters", default=None) is not None:
                updated_state["objective_parameters"] = self.get_value(self.options, "objective_parameters", default=None)
            if state_result is not None:
                updated_state["state_result"] = state_result
            if initial_guess is not None:
                updated_state["initial_guess"] = initial_guess
            elif state_result is not None:
                updated_state["initial_guess"] = state_result
            updated_state["state_solution_valid"] = state_result is not None
            updated_state["adjoint_solution_valid"] = False
            updated_state["gradient_solution_valid"] = False
            updated_state["scalar_product_valid"] = False
            return updated_state
        return {
            "mesh": accepted_mesh,
            "objective": accepted_objective,
            "current_objective": accepted_objective,
            "objective_parameters": objective_parameters if objective_parameters is not None else self.get_value(self.options, "objective_parameters", default=None),
            "trial_state": accepted_state,
            "state_result": state_result,
            "initial_guess": initial_guess if initial_guess is not None else state_result,
            "state_solution_valid": state_result is not None,
            "adjoint_solution_valid": False,
            "gradient_solution_valid": False,
            "scalar_product_valid": False,
        }

    def _invalidate_component_memory(self) -> None:
        """Invalidate cached component state after mesh updates."""
        self._component_method_cache.clear()
        for component in (self.state_solver, self.adjoint_solver, self.geometry_gradient_assembler):
            if component is None:
                continue
            if hasattr(component, "has_solution"):
                component.has_solution = False
            for name in ("invalidate_cache", "reset_cache", "reset", "clear"):
                method = getattr(component, name, None)
                if callable(method):
                    method()
                    break
        for name in ("update_scalar_product", "rebuild_scalar_product"):
            method = getattr(self.geometry_gradient_assembler, name, None)
            if callable(method):
                method()
                break

    def record_iteration_log(
        self,
        iteration: int,
        current_objective: float | None,
        trial_objective: float | None,
        step_size: float | None,
        accepted: bool | None,
        remeshed: bool = False,
        quality_info: Any = None,
    ) -> OptimizationHistoryEntry:
        """Record one iteration entry."""

        return OptimizationHistoryEntry(
            iteration=iteration,
            current_objective=current_objective,
            trial_objective=trial_objective,
            step_size=step_size,
            accepted=accepted,
            remeshed=remeshed,
            quality_info=quality_info,
        )

    def output_intermediate_result(self, result: OptimizationResult) -> None:
        """Hook for intermediate-result output."""
        return None


def check_one_step_optimization_finite_difference(
    optimizer: ShapeOptimizer,
    current_state: Any,
    objective_fn: Any,
    perturbation_fn: Any,
    step_size: float = 1.0e-6,
) -> OneStepOptimizationCheckResult:
    """Check one optimization step by finite differences."""
    current_objective = float(objective_fn(current_state))
    step_result = optimizer.step(current_state)

    trial_objective = step_result.get("trial_objective")
    if trial_objective is None:
        trial_state = step_result.get("trial_state", current_state)
        trial_objective = objective_fn(trial_state)
    trial_objective = float(trial_objective)

    geometry_gradient_result = step_result.get("geometry_gradient_result")
    descent_direction = optimizer.get_value(geometry_gradient_result, "descent_direction")
    plus_state = perturbation_fn(current_state, descent_direction, step_size)
    minus_state = perturbation_fn(current_state, descent_direction, -step_size)
    finite_difference = (
        float(objective_fn(plus_state)) - float(objective_fn(minus_state))
    ) / (2.0 * step_size)

    return OneStepOptimizationCheckResult(
        current_objective=current_objective,
        trial_objective=trial_objective,
        finite_difference=finite_difference,
        objective_decrease=current_objective - trial_objective,
        accepted=bool(step_result.get("accepted")),
    )

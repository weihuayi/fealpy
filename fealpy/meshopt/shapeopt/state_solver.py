"""Thin state-solver adapter for cashocs-style shape optimization."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any

from fealpy.backend import backend_manager as bm
from fealpy.solver import spsolve

from .benchmark_common import _write_vtu_frame, get_value
from fealpy.cfd.stationary_incompressible_stokes_lfem_model import StationaryIncompressibleStokesLFEMModel
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel


@dataclass(slots=True)
class StateSolveResult:
    velocity: Any = None
    pressure: Any = None
    system_matrix: Any = None
    rhs: Any = None
    mesh: Any = None
    state_vector: Any = None
    velocity_space: Any = None
    pressure_space: Any = None
    converged: bool | None = None
    nonlinear_iterations: int | None = None
    final_residual_u: float | None = None
    final_residual_p: float | None = None
    velocity_dirichlet_threshold: Any = None
    pressure_dirichlet_threshold: Any = None
    equation_kind: str | None = None


def _write_state_vtu_if_requested(mesh: Any, state_result: StateSolveResult, options: Any = None) -> None:
    vtu_path = get_value(options, "debug_vtu_path", "state_vtu_path")
    if vtu_path is not None:
        _write_vtu_frame(Path(vtu_path), mesh, state_result)


def _attach_solver_metadata(state_result: StateSolveResult, solver: Any, mesh: Any) -> StateSolveResult:
    if state_result.mesh is None:
        state_result.mesh = mesh
    fem = getattr(solver, "fem", None)
    if fem is not None:
        if state_result.velocity_space is None:
            state_result.velocity_space = getattr(fem, "uspace", getattr(fem, "velocity_space", None))
        if state_result.pressure_space is None:
            state_result.pressure_space = getattr(fem, "pspace", getattr(fem, "pressure_space", None))
    pde = getattr(solver, "pde", None)
    if state_result.velocity_dirichlet_threshold is None:
        state_result.velocity_dirichlet_threshold = get_value(
            pde,"velocity_dirichlet_threshold",
            default=getattr(fem, "velocity_dirichlet_threshold", None) if fem is not None else None,
        )
    if state_result.pressure_dirichlet_threshold is None:
        state_result.pressure_dirichlet_threshold = get_value(
            pde,"pressure_dirichlet_threshold",
            default=getattr(fem, "pressure_dirichlet_threshold", None) if fem is not None else None,
        )
    return state_result


def _build_fem_update_guess(fem: Any, initial_guess: Any = None) -> Any:
    guessed_velocity = get_value(initial_guess, "velocity", "state_velocity", "u", default=None)
    if guessed_velocity is not None:
        return guessed_velocity
    velocity_space = getattr(fem, "uspace", None)
    if velocity_space is None:
        return None
    return velocity_space.function()

def _build_ns_initial_guess(fem, pde, initial_guess=None):
    guessed_velocity = get_value(initial_guess, "velocity", "state_velocity", "u", default=None)
    if guessed_velocity is not None:
        return guessed_velocity

    velocity_space = getattr(fem, "uspace", None)
    if velocity_space is None:
        return None

    u0 = velocity_space.function()
    u0[:] = velocity_space.interpolate(pde.velocity_dirichlet)
    return u0


def _normalize_ns_method_name(method_name: Any) -> str:
    if method_name is None:
        return "Ossen"
    normalized = str(method_name).strip()
    aliases = {
        "oseen": "Ossen",
        "ossen": "Ossen",
        "stokes": "Stokes",
        "newton": "Newton",
    }
    return aliases.get(normalized.lower(), normalized)


def _get_ns_solver_option(options: Any, fluid_model: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = get_value(options, name, default=None)
        if value is not None:
            return value
        value = get_value(getattr(fluid_model, "options", None), name, default=None)
        if value is not None:
            return value
    return default


def _clone_function(space: Any, values: Any) -> Any:
    cloned = space.function()
    cloned[:] = values[:]
    return cloned


def _configure_ns_constitutive(fluid_model: Any) -> None:
    equation = getattr(fluid_model, "equation", None)
    setter = getattr(equation, "set_constitutive", None)
    if callable(setter):
        setter("simplified_laplacian")


def _build_ns_stokes_warm_start(
    fluid_model: Any,
    selected_method: str,
    initial_guess: Any = None,
) -> tuple[Any, Any] | None:
    if initial_guess is not None:
        return None

    fluid_model.method.set("Stokes")
    warm_fem = fluid_model.method()
    BForm, LForm = fluid_model.linear_system()
    update = getattr(warm_fem, "update", None)
    if callable(update):
        try:
            update_parameters = signature(update).parameters
        except (TypeError, ValueError):
            update_parameters = None
        if update_parameters and len(update_parameters) > 0:
            update(_build_fem_update_guess(warm_fem))
        else:
            update()
    matrix = BForm.assembly()
    rhs = LForm.assembly()
    matrix, rhs = warm_fem.apply_bc(matrix, rhs, getattr(fluid_model, "pde", fluid_model))
    x = spsolve(matrix, rhs, solver="mumps")
    ugdof = warm_fem.uspace.number_of_global_dofs()
    pgdof = warm_fem.pspace.number_of_global_dofs()
    velocity = warm_fem.uspace.function()
    pressure = warm_fem.pspace.function()
    velocity[:] = x[:ugdof]
    pressure[:] = x[ugdof : ugdof + pgdof]
    fluid_model.method.set(selected_method)
    fluid_model.method()
    return velocity, pressure

def stokes_state_solver(fluid_model:Any,initial_guess: Any = None) -> StateSolveResult:
    fem = get_value(fluid_model, "fem", default=None)
    pde = getattr(fluid_model, "pde", fluid_model)
    if getattr(pde, "velocity_dirichlet_threshold", None) is None:
        velocity_boundary = getattr(pde, "is_velocity_boundary", None)
        if callable(velocity_boundary):
            try:
                pde.velocity_dirichlet_threshold = velocity_boundary(fem.uspace)
            except (TypeError, ValueError, AttributeError):
                pde.velocity_dirichlet_threshold = velocity_boundary
    if getattr(pde, "pressure_dirichlet_threshold", None) is None:
        pressure_boundary = getattr(pde, "is_pressure_boundary", None)
        if callable(pressure_boundary):
            try:
                pde.pressure_dirichlet_threshold = pressure_boundary(fem.pspace)
            except (TypeError, ValueError, AttributeError):
                pde.pressure_dirichlet_threshold = pressure_boundary
    BForm, LForm = fluid_model.linear_system()
    update = getattr(fem, "update", None)
    if callable(update):
        try:
            update_parameters = signature(update).parameters
        except (TypeError, ValueError):
            update_parameters = None
        if update_parameters and len(update_parameters) > 0:
            update(_build_fem_update_guess(fem, initial_guess=initial_guess))
        else:
            update()

    matrix = BForm.assembly()
    rhs = LForm.assembly()
    matrix, rhs = fem.apply_bc(matrix, rhs, getattr(fluid_model, "pde", fluid_model))
    x = spsolve(matrix, rhs, solver='mumps')
    ugdof = fem.uspace.number_of_global_dofs()
    pgdof = fem.pspace.number_of_global_dofs()
    velocity = fem.uspace.function()
    pressure = fem.pspace.function()
    velocity[:] = x[:ugdof]
    pressure[:] = x[ugdof : ugdof + pgdof]
    return StateSolveResult(
        velocity=velocity,
        pressure=pressure,
        system_matrix=matrix,
        rhs=rhs,
        mesh= getattr(fluid_model, "mesh", None),
        velocity_space=fem.uspace,
        pressure_space=fem.pspace,
        velocity_dirichlet_threshold=get_value(
            pde,"velocity_dirichlet_threshold",
            default=getattr(fem, "velocity_dirichlet_threshold", None),
        ),
        pressure_dirichlet_threshold=get_value(
            pde,"pressure_dirichlet_threshold",
            default=getattr(fem, "pressure_dirichlet_threshold", None),
        ),
    )

def ns_state_solver(fluid_model:Any,initial_guess: Any = None) -> StateSolveResult:
    selected_method = _normalize_ns_method_name(
        _get_ns_solver_option(
            None,
            fluid_model,
            "state_method",
            "ns_method",
            "method",
            default="Ossen",
        )
    )
    _configure_ns_constitutive(fluid_model)
    fluid_model.method.set(selected_method)
    fluid_model.equation.pressure_neumann = True
    fem = fluid_model.method()
    pde = getattr(fluid_model, "pde", fluid_model)
    if getattr(pde, "velocity_dirichlet_threshold", None) is None:
        velocity_boundary = getattr(pde, "is_velocity_boundary", None)
        if callable(velocity_boundary):
            try:
                pde.velocity_dirichlet_threshold = velocity_boundary(fem.uspace)
            except (TypeError, ValueError, AttributeError):
                pde.velocity_dirichlet_threshold = velocity_boundary
    if getattr(pde, "pressure_dirichlet_threshold", None) is None:
        pressure_boundary = getattr(pde, "is_pressure_boundary", None)
        if callable(pressure_boundary):
            try:
                pde.pressure_dirichlet_threshold = pressure_boundary(fem.pspace)
            except (TypeError, ValueError, AttributeError):
                pde.pressure_dirichlet_threshold = pressure_boundary
    u0 = _build_ns_initial_guess(fem, fluid_model.pde, initial_guess)
    warm_start = _build_ns_stokes_warm_start(fluid_model, selected_method, initial_guess=initial_guess)
    if warm_start is not None:
        uh0, ph0 = warm_start
        fem = fluid_model.method()
    else:
        uh0 = u0
        ph0 = fem.pspace.function()
    mesh = getattr(fluid_model, "mesh", None)
    tol = float(_get_ns_solver_option(None, fluid_model, "ns_iteration_tol", "tol", default=1.0e-6))
    max_iterations = int(_get_ns_solver_option(None, fluid_model, "ns_iteration_max_steps", "maxstep", default=40))
    alpha = float(_get_ns_solver_option(None, fluid_model, "ns_relaxation", default=0.5))
    min_alpha = float(_get_ns_solver_option(None, fluid_model, "ns_min_relaxation", default=0.1))
    max_alpha = float(_get_ns_solver_option(None, fluid_model, "ns_max_relaxation", default=1.0))
    previous_residual = None
    converged = False
    uh1 = _clone_function(fem.uspace, uh0)
    ph1 = _clone_function(fem.pspace, ph0)
    res_u = float("inf")
    res_p = float("inf")

    for i in range(max_iterations):
        uh1, ph1 = fluid_model.run['one_step'](uh0)

        res_u = mesh.error(uh0, uh1)
        res_p = mesh.error(ph0, ph1)
        residual = max(float(res_u), float(res_p))
        print(
            f"{selected_method} iteration {i+1}, residual_u={res_u:.2e}, "
            f"residual_p={res_p:.2e}, alpha={alpha:.2f}"
        )

        if residual <= tol:
            converged = True
            print(f"Converged at iteration {i+1}")
            break

        if previous_residual is not None:
            if residual > previous_residual * 1.05:
                alpha = max(min_alpha, 0.5 * alpha)
            else:
                alpha = min(max_alpha, alpha * 1.2)

        uh0[:] = (1.0 - alpha) * uh0[:] + alpha * uh1[:]
        ph0[:] = (1.0 - alpha) * ph0[:] + alpha * ph1[:]

        previous_residual = residual

    final_velocity = uh1 if converged else uh0
    final_pressure = ph1 if converged else ph0

    BForm, LForm = fluid_model.linear_system()
    fem.update(final_velocity)
    A = BForm.assembly()
    b = LForm.assembly()
    A, b = fem.apply_bc(A, b, pde)
    if fluid_model.equation.pressure_neumann == True:
        A, b = fem.lagrange_multiplier(A, b, c = pde.pressure_integral_target())

    return StateSolveResult(
        velocity=_clone_function(fem.uspace, final_velocity),
        pressure=_clone_function(fem.pspace, final_pressure),
        system_matrix=A,
        rhs=b,
        mesh= getattr(fluid_model, "mesh", None),
        velocity_space=getattr(fluid_model.fem, "uspace", None),
        pressure_space=getattr(fluid_model.fem, "pspace", None),
        velocity_dirichlet_threshold=get_value(
            pde,"velocity_dirichlet_threshold",
            default=getattr(fem, "velocity_dirichlet_threshold", None),
        ),
        pressure_dirichlet_threshold=get_value(
            pde,"pressure_dirichlet_threshold",
            default=getattr(fem, "pressure_dirichlet_threshold", None),
        ),
        converged=converged,
        nonlinear_iterations=i + 1,
        final_residual_u=float(res_u),
        final_residual_p=float(res_p),
        equation_kind="ns",
    )
    

def solve_state_system(
    mesh: Any,
    fluid_model: Any,
    geometry_contract: Any,
    initial_guess: Any = None,
    options: Any = None,
) -> StateSolveResult:
    
    fluid_model.update_mesh(mesh)
    if isinstance(fluid_model, StationaryIncompressibleStokesLFEMModel):
        state_result = stokes_state_solver(fluid_model, initial_guess=initial_guess)
    elif isinstance(fluid_model, StationaryIncompressibleNSLFEMModel):
        state_result = ns_state_solver(fluid_model, initial_guess=initial_guess)
    _write_state_vtu_if_requested(mesh, state_result, options)
    return _attach_solver_metadata(state_result, fluid_model, mesh)

"""Algorithm-specific pressure systems for collocated incompressible flow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias

from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor
from fealpy.typing import TensorLike

from .collocated_pressure_equation import CollocatedPressureEquation
from .collocated_linear_solvers import (
    CollocatedNSLinearSolvers,
    LinearSystemSolver,
)
from .fvm_linear_solver import (
    LinearSolveDiagnostics,
    LinearSolveResult,
)
from .solver_diagnostics import (
    EquationResidual,
    equation_residual_converged,
    normalized_equation_residual,
)

if TYPE_CHECKING:
    from .collocated_boundary_conditions import (
        ResolvedPressureSystemBoundary,
    )


class PressureClosureKind(StrEnum):
    """Algebraic closure selected before a pressure system is constructed."""

    DIRICHLET = "dirichlet"
    NULLSPACE = "nullspace"
    GAUGE = "gauge"


@dataclass(frozen=True)
class CollocatedPressureSystemControls:
    """Choose the algebraic closure for a pure-Neumann pressure system."""

    pure_neumann_closure: PressureClosureKind = (
        PressureClosureKind.NULLSPACE
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.pure_neumann_closure,
                PressureClosureKind,
            )
            or self.pure_neumann_closure not in {
                PressureClosureKind.NULLSPACE,
                PressureClosureKind.GAUGE,
            }
        ):
            raise ValueError(
                "pure_neumann_closure must be "
                "PressureClosureKind.NULLSPACE or "
                "PressureClosureKind.GAUGE."
            )


@dataclass(frozen=True)
class PressureCorrectionResult:
    """Call-local SIMPLE pressure-correction solve result."""

    pressure_correction: TensorLike
    nonorthogonal_iterations: int
    nonorthogonal_residual: EquationResidual | None
    nonorthogonal_relative_update: float
    linear_solves: tuple[LinearSolveDiagnostics, ...]


@dataclass(frozen=True)
class PressureFluxParts:
    """PISO pressure-flux decomposition at the converged pressure state."""

    orthogonal_flux: TensorLike
    cross_flux: TensorLike
    boundary_pressure_flux: TensorLike


@dataclass(frozen=True)
class PisoPressureResult:
    """Call-local PISO pressure-state solve and its matching face flux."""

    pressure: TensorLike
    pressure_flux: TensorLike
    flux_parts: PressureFluxParts
    face_response_coefficient: TensorLike
    nonorthogonal_iterations: int
    nonorthogonal_residual: EquationResidual | None
    nonorthogonal_relative_update: float
    linear_solves: tuple[LinearSolveDiagnostics, ...]


@dataclass(frozen=True)
class NonorthogonalPressureResult:
    """Shared pressure state produced by one explicit cross-flux loop."""

    pressure: TensorLike
    cross_flux: TensorLike
    iterations: int
    residual: EquationResidual | None
    relative_update: float
    linear_solves: tuple[LinearSolveDiagnostics, ...]


class DirichletPressureSystemClosure:
    def __init__(
        self,
        boundary: ResolvedPressureSystemBoundary,
        equation: CollocatedPressureEquation,
        linear_solver: LinearSystemSolver,
    ) -> None:
        self.boundary = boundary
        self.equation = equation
        self.linear_solver = linear_solver

    def matrix(self, coefficient: TensorLike) -> CSRTensor:
        base = self.equation.diffusion_matrix(coefficient)
        return self.boundary.dirichlet_operator.apply_diffusion_matrix(
            base,
            coef=coefficient,
            components=1,
        )

    def rhs(
        self,
        base_rhs: TensorLike,
        cross_rhs: TensorLike,
        coefficient: TensorLike,
    ) -> TensorLike:
        return self.boundary.dirichlet_operator.apply_diffusion_rhs(
            base_rhs + cross_rhs,
            coef=coefficient,
            components=1,
        )

    def encode(self, pressure: TensorLike) -> TensorLike:
        return pressure

    def decode(self, solution: TensorLike) -> TensorLike:
        return solution

    def solve(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
    ) -> LinearSolveResult:
        return self.linear_solver.solve(
            matrix,
            rhs,
        )


class NullspacePressureSystemClosure:
    def __init__(
        self,
        equation: CollocatedPressureEquation,
        linear_solver: LinearSystemSolver,
    ) -> None:
        self.equation = equation
        self.linear_solver = linear_solver

    def matrix(self, coefficient: TensorLike) -> CSRTensor:
        return self.equation.diffusion_matrix(coefficient)

    def rhs(
        self,
        base_rhs: TensorLike,
        cross_rhs: TensorLike,
        coefficient: TensorLike,
    ) -> TensorLike:
        return self.equation.project_rhs_to_range(base_rhs + cross_rhs)

    def encode(self, pressure: TensorLike) -> TensorLike:
        return pressure

    def decode(self, solution: TensorLike) -> TensorLike:
        return self.equation.zero_mean_pressure(solution)

    def solve(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
    ) -> LinearSolveResult:
        result = self.linear_solver.solve(matrix, rhs)
        return LinearSolveResult(
            solution=self.decode(result.solution),
            diagnostics=result.diagnostics,
        )


class GaugePressureSystemClosure:
    def __init__(
        self,
        equation: CollocatedPressureEquation,
        linear_solver: LinearSystemSolver,
    ) -> None:
        self.equation = equation
        self.linear_solver = linear_solver

    def matrix(self, coefficient: TensorLike) -> CSRTensor:
        return self.equation.gauge_matrix(coefficient)

    def rhs(
        self,
        base_rhs: TensorLike,
        cross_rhs: TensorLike,
        coefficient: TensorLike,
    ) -> TensorLike:
        return bm.concatenate(
            [
                base_rhs + cross_rhs,
                bm.zeros_like(base_rhs[:1]),
            ],
            axis=0,
        )

    def encode(self, pressure: TensorLike) -> TensorLike:
        return bm.concatenate(
            [pressure, bm.zeros_like(pressure[:1])],
            axis=0,
        )

    def decode(self, solution: TensorLike) -> TensorLike:
        return solution[:-1]

    def solve(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
    ) -> LinearSolveResult:
        result = self.linear_solver.solve(matrix, rhs)
        return LinearSolveResult(
            solution=self.decode(result.solution),
            diagnostics=result.diagnostics,
        )


PressureSystemClosure: TypeAlias = (
    DirichletPressureSystemClosure
    | NullspacePressureSystemClosure
    | GaugePressureSystemClosure
)


def resolve_pressure_system_closure(
    boundary: ResolvedPressureSystemBoundary,
    pressure_equation: CollocatedPressureEquation,
    linear_solvers: CollocatedNSLinearSolvers,
) -> PressureSystemClosure:
    """Bind one resolved closure to its concrete linear solver."""
    closure = boundary.closure
    if closure is PressureClosureKind.DIRICHLET:
        return DirichletPressureSystemClosure(
            boundary,
            pressure_equation,
            linear_solvers.pressure_dirichlet,
        )
    if closure is PressureClosureKind.NULLSPACE:
        return NullspacePressureSystemClosure(
            pressure_equation,
            linear_solvers.pressure_nullspace,
        )
    if closure is PressureClosureKind.GAUGE:
        return GaugePressureSystemClosure(
            pressure_equation,
            linear_solvers.pressure_gauge,
        )
    raise TypeError("unknown resolved pressure closure.")


def iterate_nonorthogonal_pressure(
    *,
    equation: CollocatedPressureEquation,
    boundary: ResolvedPressureSystemBoundary,
    closure: PressureSystemClosure,
    matrix: CSRTensor,
    base_rhs: TensorLike,
    coefficient: TensorLike,
    initial_cross_rhs: TensorLike,
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> NonorthogonalPressureResult:
    """Solve one pressure system and its explicit cross-flux corrections."""

    def solve_with_cross_rhs(
        cross_rhs: TensorLike,
    ) -> LinearSolveResult:
        rhs = closure.rhs(base_rhs, cross_rhs, coefficient)
        return closure.solve(matrix, rhs)

    initial_solve = solve_with_cross_rhs(initial_cross_rhs)
    pressure = initial_solve.solution
    linear_solves = [initial_solve.diagnostics]
    cross_flux = bm.zeros(
        equation.discretization.NF,
        dtype=base_rhs.dtype,
        device=bm.get_device(base_rhs),
    )
    iterations = 0
    residual = None
    relative_update = 0.0
    if max_iterations == 0:
        return NonorthogonalPressureResult(
            pressure=pressure,
            cross_flux=cross_flux,
            iterations=iterations,
            residual=residual,
            relative_update=relative_update,
            linear_solves=tuple(linear_solves),
        )

    for correction in range(max_iterations + 1):
        gradient = boundary.gradient.cell_gradient(pressure)
        cross_flux = equation.nonorthogonal_cross_flux(
            pressure,
            coefficient,
            pressure_gradient=gradient,
            gradient_boundary=boundary.gradient.boundary,
            interpolation_method=equation.response_interpolation,
        )
        cross_rhs = equation.divergence_from_flux(cross_flux)
        rhs = closure.rhs(base_rhs, cross_rhs, coefficient)
        residual = normalized_equation_residual(
            matrix @ closure.encode(pressure),
            rhs,
        )
        if equation_residual_converged(
            residual,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            return NonorthogonalPressureResult(
                pressure=pressure,
                cross_flux=cross_flux,
                iterations=iterations,
                residual=residual,
                relative_update=relative_update,
                linear_solves=tuple(linear_solves),
            )
        if correction == max_iterations:
            break
        previous = pressure
        solve_result = solve_with_cross_rhs(cross_rhs)
        pressure = solve_result.solution
        linear_solves.append(solve_result.diagnostics)
        update = float(
            bm.to_numpy(bm.linalg.norm(pressure - previous))
        )
        scale = float(bm.to_numpy(bm.linalg.norm(pressure)))
        relative_update = update / max(scale, 1.0e-30)
        iterations = correction + 1

    raise RuntimeError(
        "pressure non-orthogonal correction did not converge before "
        "the configured maximum iterations"
    )


class SimplePressureCorrectionSystem:
    """Solve the SIMPLE pressure-correction closure and nonorthogonal loop."""

    def __init__(
        self,
        *,
        pressure_equation: CollocatedPressureEquation,
        boundary: ResolvedPressureSystemBoundary,
        closure: PressureSystemClosure,
        pressure_nonorthogonal_max_iterations: int,
        pressure_nonorthogonal_rtol: float,
        pressure_nonorthogonal_atol: float,
    ) -> None:
        self.pressure_equation = pressure_equation
        self.boundary = boundary
        self.closure = closure
        self.pressure_nonorthogonal_max_iterations = int(
            pressure_nonorthogonal_max_iterations
        )
        self.pressure_nonorthogonal_rtol = float(
            pressure_nonorthogonal_rtol
        )
        self.pressure_nonorthogonal_atol = float(
            pressure_nonorthogonal_atol
        )

    def correction_flux(
        self,
        pressure_correction: TensorLike,
        response_coefficient: TensorLike,
        *,
        pressure_gradient: TensorLike,
    ) -> TensorLike:
        equation = self.pressure_equation
        flux = equation.orthogonal_flux(
            pressure_correction,
            response_coefficient,
        )
        flux = flux - equation.nonorthogonal_cross_flux(
            pressure_correction,
            response_coefficient,
            pressure_gradient=pressure_gradient,
            gradient_boundary=self.boundary.gradient.boundary,
            interpolation_method=equation.response_interpolation,
        )
        return equation.add_dirichlet_flux(
            flux,
            pressure_correction,
            response_coefficient,
            self.boundary.dirichlet_operator.faces,
            self.boundary.dirichlet_operator.values,
        )

    def solve(
        self,
        face_velocity: TensorLike,
        response_diagonal: TensorLike,
    ) -> PressureCorrectionResult:
        equation = self.pressure_equation
        coefficient = equation.face_response_coefficient(
            response_diagonal
        )
        base_rhs = -equation.divergence_from_face_velocity(
            face_velocity
        )
        matrix = self.closure.matrix(coefficient)
        state = iterate_nonorthogonal_pressure(
            equation=equation,
            boundary=self.boundary,
            closure=self.closure,
            matrix=matrix,
            base_rhs=base_rhs,
            coefficient=coefficient,
            initial_cross_rhs=bm.zeros_like(base_rhs),
            max_iterations=(
                self.pressure_nonorthogonal_max_iterations
            ),
            relative_tolerance=self.pressure_nonorthogonal_rtol,
            absolute_tolerance=self.pressure_nonorthogonal_atol,
        )
        return PressureCorrectionResult(
            pressure_correction=state.pressure,
            nonorthogonal_iterations=state.iterations,
            nonorthogonal_residual=state.residual,
            nonorthogonal_relative_update=state.relative_update,
            linear_solves=state.linear_solves,
        )


class PisoPressureSystem:
    """Solve the PISO pressure state and return the matching pressure flux."""

    def __init__(
        self,
        *,
        pressure_equation: CollocatedPressureEquation,
        boundary: ResolvedPressureSystemBoundary,
        closure: PressureSystemClosure,
        pressure_nonorthogonal_max_iterations: int,
        pressure_nonorthogonal_rtol: float,
        pressure_nonorthogonal_atol: float,
    ) -> None:
        self.pressure_equation = pressure_equation
        self.boundary = boundary
        self.closure = closure
        self.pressure_nonorthogonal_max_iterations = int(
            pressure_nonorthogonal_max_iterations
        )
        self.pressure_nonorthogonal_rtol = float(
            pressure_nonorthogonal_rtol
        )
        self.pressure_nonorthogonal_atol = float(
            pressure_nonorthogonal_atol
        )

    def solve(
        self,
        rhs: TensorLike,
        response_diagonal: TensorLike,
        *,
        initial_pressure_state: TensorLike,
    ) -> PisoPressureResult:
        equation = self.pressure_equation
        coefficient = equation.face_response_coefficient(
            response_diagonal
        )
        matrix = self.closure.matrix(coefficient)
        cross_rhs = bm.zeros_like(rhs)
        max_iterations = self.pressure_nonorthogonal_max_iterations
        if max_iterations > 0:
            gradient = self.boundary.gradient.cell_gradient(
                initial_pressure_state
            )
            cross_flux = equation.nonorthogonal_cross_flux(
                initial_pressure_state,
                coefficient,
                pressure_gradient=gradient,
                gradient_boundary=self.boundary.gradient.boundary,
                interpolation_method=equation.response_interpolation,
            )
            cross_rhs = equation.divergence_from_flux(cross_flux)
        state = iterate_nonorthogonal_pressure(
            equation=equation,
            boundary=self.boundary,
            closure=self.closure,
            matrix=matrix,
            base_rhs=rhs,
            coefficient=coefficient,
            initial_cross_rhs=cross_rhs,
            max_iterations=max_iterations,
            relative_tolerance=self.pressure_nonorthogonal_rtol,
            absolute_tolerance=self.pressure_nonorthogonal_atol,
        )
        pressure = state.pressure
        cross_flux = state.cross_flux

        orthogonal_flux = equation.orthogonal_flux(
            pressure,
            coefficient,
        )
        boundary_flux = equation.add_dirichlet_flux(
            bm.zeros_like(orthogonal_flux),
            pressure,
            coefficient,
            self.boundary.dirichlet_operator.faces,
            self.boundary.dirichlet_operator.values,
        )
        pressure_flux = equation.add_dirichlet_flux(
            orthogonal_flux - cross_flux,
            pressure,
            coefficient,
            self.boundary.dirichlet_operator.faces,
            self.boundary.dirichlet_operator.values,
        )
        return PisoPressureResult(
            pressure=pressure,
            pressure_flux=pressure_flux,
            flux_parts=PressureFluxParts(
                orthogonal_flux=orthogonal_flux,
                cross_flux=cross_flux,
                boundary_pressure_flux=boundary_flux,
            ),
            face_response_coefficient=coefficient,
            nonorthogonal_iterations=state.iterations,
            nonorthogonal_residual=state.residual,
            nonorthogonal_relative_update=state.relative_update,
            linear_solves=state.linear_solves,
        )


__all__ = [
    "CollocatedPressureSystemControls",
    "DirichletPressureSystemClosure",
    "GaugePressureSystemClosure",
    "NullspacePressureSystemClosure",
    "NonorthogonalPressureResult",
    "PressureClosureKind",
    "resolve_pressure_system_closure",
    "PisoPressureResult",
    "PisoPressureSystem",
    "PressureCorrectionResult",
    "PressureFluxParts",
    "SimplePressureCorrectionSystem",
    "iterate_nonorthogonal_pressure",
]

"""Collocated PISO solver core for transient incompressible Navier-Stokes."""

from collections.abc import Callable
from typing import TypeAlias

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor

from .collocated_boundary_conditions import ResolvedPisoBoundaryConditions
from .collocated_discretization import CollocatedDiscretization
from .collocated_momentum_equation import (
    CollocatedMomentumSpatialOperator,
    ComponentMomentumAlgebra,
    TransientMomentumEquation,
)
from .collocated_pressure_system import (
    PisoPressureSystem,
    resolve_pressure_system_closure,
)
from .collocated_pressure_equation import CollocatedPressureEquation
from .collocated_linear_solvers import CollocatedNSLinearSolvers
from .solver_controls import PisoSolverControls, positive_scalar
from .collocated_face_velocity_reconstruct import RhieChowInterpolation
from .collocated_spatial_face_velocity import CollocatedSpatialFaceVelocity
from .collocated_velocity_pressure_coupling import (
    correct_cell_velocity,
    remove_pressure_response,
)
from .face_flux_reconstruct import FaceFluxReconstruct
from .piso_result import (
    PisoCorrectorDiagnostic,
    PisoPressureCorrectionDiagnostics,
    PisoPressureCorrectionStepResult,
    PisoSnapshot,
    PisoSolveResult,
)
from .solver_diagnostics import linf_norm

PisoSourceFunction: TypeAlias = Callable[
    [TensorLike, float],
    TensorLike,
]
PisoSnapshotCallback: TypeAlias = Callable[[PisoSnapshot], None]
PisoCorrectorCallback: TypeAlias = Callable[
    [PisoCorrectorDiagnostic],
    None,
]


class CollocatedPisoSolver:
    """Algorithm core for transient collocated PISO solves."""

    def __init__(
        self,
        *,
        diffusion_coef: float,
        convection_coef: float,
        source: PisoSourceFunction,
        boundary_conditions: ResolvedPisoBoundaryConditions,
        controls: PisoSolverControls,
        linear_solvers: CollocatedNSLinearSolvers,
    ) -> None:
        if not isinstance(controls, PisoSolverControls):
            raise TypeError("controls must be a PisoSolverControls.")
        if not isinstance(
            linear_solvers,
            CollocatedNSLinearSolvers,
        ):
            raise TypeError(
                "linear_solvers must be "
                "CollocatedNSLinearSolvers."
            )
        self.controls = controls
        diffusion_coef = positive_scalar(diffusion_coef, "diffusion_coef")
        convection_coef = positive_scalar(convection_coef, "convection_coef")
        if not isinstance(
            boundary_conditions,
            ResolvedPisoBoundaryConditions,
        ):
            raise TypeError(
                "CollocatedPisoSolver requires "
                "ResolvedPisoBoundaryConditions."
            )
        physical = boundary_conditions.physical
        discretization = CollocatedDiscretization(
            geometry=physical.geometry,
        )
        self.discretization = discretization
        self.pressure_gradient = physical.pressure_state.gradient
        self.rhie_chow = RhieChowInterpolation(
            discretization.geometry,
            boundary_conditions.rhie_chow,
        )
        face_flux_reconstruct = FaceFluxReconstruct(
            geometry=discretization.geometry,
            method="none",
        )
        self.spatial_face_velocity = CollocatedSpatialFaceVelocity(
            discretization=discretization,
            boundary=physical.velocity,
            scheme="interpolated",
            interpolation=controls.rhie_chow_velocity_interpolation,
            face_flux_reconstruct=face_flux_reconstruct,
        )
        momentum_spatial = CollocatedMomentumSpatialOperator(
            discretization=discretization,
            momentum_boundary=physical.momentum,
            diffusion_coef=diffusion_coef,
            convection_coef=convection_coef,
            diffusion_method=self.controls.diffusion_method,
            diffusion_nonorthogonal_eps=(
                self.controls.diffusion_nonorthogonal_eps
            ),
            face_interpolation=(
                self.controls.momentum_face_interpolation
            ),
        )
        momentum_algebra = ComponentMomentumAlgebra(
            discretization=discretization,
            linear_solver=linear_solvers.momentum,
        )
        self.momentum = TransientMomentumEquation(
            spatial_operator=momentum_spatial,
            algebra=momentum_algebra,
            source=source,
            density=convection_coef,
            time_step=self.controls.tau,
            momentum_nonorthogonal_max_iterations=(
                self.controls.momentum_nonorthogonal_max_iterations
            ),
            momentum_nonorthogonal_rtol=(
                self.controls.momentum_nonorthogonal_rtol
            ),
            momentum_nonorthogonal_atol=(
                self.controls.momentum_nonorthogonal_atol
            ),
        )
        self.pressure_equation = CollocatedPressureEquation(
            discretization=discretization,
            diffusion_method=self.controls.diffusion_method,
            diffusion_nonorthogonal_eps=(
                self.controls.diffusion_nonorthogonal_eps
            ),
            response_interpolation=(
                self.controls.pressure_response_interpolation
            ),
        )
        pressure_boundary = boundary_conditions.pressure_corrector
        pressure_closure = resolve_pressure_system_closure(
            pressure_boundary,
            self.pressure_equation,
            linear_solvers,
        )
        self.pressure_system = PisoPressureSystem(
            pressure_equation=self.pressure_equation,
            boundary=pressure_boundary,
            closure=pressure_closure,
            pressure_nonorthogonal_max_iterations=(
                self.controls.pressure_nonorthogonal_max_iterations
            ),
            pressure_nonorthogonal_rtol=(
                self.controls.pressure_nonorthogonal_rtol
            ),
            pressure_nonorthogonal_atol=(
                self.controls.pressure_nonorthogonal_atol
            ),
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.discretization.NC} cells\n"
            f"  Time steps: {self.controls.time_steps}\n"
            f"  PISO correctors: {self.controls.n_correctors}\n"
            f"  Momentum nonorthogonal corrections: "
            f"{self.controls.momentum_nonorthogonal_max_iterations}\n"
            f"  Pressure nonorthogonal corrections: "
            f"{self.controls.pressure_nonorthogonal_max_iterations}\n"
        )

    def rhie_chow_face_velocity(
        self,
        cell_velocity: TensorLike,
        pressure: TensorLike,
        face_response_coefficient: TensorLike,
    ) -> TensorLike:
        """Build a collocated face velocity for the current PISO substep."""
        base_face_velocity = self.spatial_face_velocity.reconstruct(
            cell_velocity
        )
        pressure_gradient = self.pressure_gradient.cell_gradient(pressure)
        face_velocity = self.rhie_chow.apply(
            base_face_velocity,
            pressure,
            face_response_coefficient,
            pressure_gradient,
        )
        return self.spatial_face_velocity.enforce_boundary(
            face_velocity
        )

    def pressure_free_flux(
        self,
        intermediate_velocity: TensorLike,
        pressure: TensorLike,
        a_p: TensorLike,
    ) -> tuple[TensorLike, TensorLike]:
        """Return pressure-free velocity and matching interpolated face flux."""
        pressure_free_velocity = remove_pressure_response(
            intermediate_velocity,
            self.pressure_gradient.cell_gradient(pressure),
            self.discretization.cell_response(a_p),
        )
        face_velocity = self.spatial_face_velocity.interpolate(
            pressure_free_velocity
        )
        return (
            pressure_free_velocity,
            self.spatial_face_velocity.compute_flux(face_velocity),
        )

    def transient_face_flux_correction(
        self,
        previous_cell_velocity: TensorLike,
        previous_face_velocity: TensorLike,
        a_p: TensorLike,
    ) -> TensorLike:
        """Return the Euler ``rAU_f * ddtCorr(U, face_flux)`` correction."""
        if not self.controls.use_transient_flux_correction:
            return bm.zeros_like(
                self.discretization.geometry.face_measure
            )

        previous_face_flux = self.spatial_face_velocity.compute_flux(
            previous_face_velocity
        )
        previous_cell_flux = self.spatial_face_velocity.compute_flux(
            self.spatial_face_velocity.interpolate(
                previous_cell_velocity
            )
        )
        flux_correction = previous_face_flux - previous_cell_flux
        denominator = bm.abs(previous_face_flux) + 1.0e-300
        limiter = 1.0 - bm.minimum(bm.abs(flux_correction) / denominator, 1.0)
        limiter = bm.where(
            self.discretization.geometry.is_boundary,
            0.0,
            limiter,
        )
        flux_correction = limiter * flux_correction
        response_coef = (
            self.pressure_equation.face_response_coefficient(a_p)
        )
        return response_coef * flux_correction / self.controls.tau

    def piso_neighbour_velocity_correction(
        self,
        corrected_velocity: TensorLike,
        predicted_velocity: TensorLike,
        momentum_matrix: CSRTensor,
        a_p: TensorLike,
    ) -> TensorLike:
        """Add the PISO neighbour-velocity compensation before correction two.

        The momentum matrix is stored on the left-hand side, so its off-diagonal
        action is the negative of ``sum_N a_PN delta_U_N`` in the control-volume
        derivation.  Therefore the explicit PISO compensation is

            -(A delta_U - diag(A) delta_U) / diag(A).
        """
        delta_u = corrected_velocity - predicted_velocity
        delta_dofs = self.discretization.cell_vector_to_dofs(delta_u)
        offdiag_delta = self.momentum.matrix_action(
            momentum_matrix,
            delta_dofs,
        )
        offdiag_delta = (
            offdiag_delta
            - self.discretization.component_cell_diagonal(a_p)
            * delta_dofs
        )
        offdiag_cell = self.discretization.dofs_to_cell_vector(
            offdiag_delta
        )
        return corrected_velocity - offdiag_cell / a_p[:, None]

    def pressure_correction_step(
        self,
        intermediate_velocity: TensorLike,
        pressure: TensorLike,
        a_p: TensorLike,
        previous_cell_velocity: TensorLike,
        previous_face_velocity: TensorLike,
        *,
        return_diagnostics: bool = False,
    ) -> PisoPressureCorrectionStepResult:
        """Perform one PISO pressure-correction step.

        The pressure equation is solved for the pressure state associated with
        the current pressure-free velocity estimate, using the pressure-state
        PISO corrector form without introducing a separate flux abstraction.
        """
        pressure_free_velocity, flux = self.pressure_free_flux(
            intermediate_velocity,
            pressure,
            a_p,
        )
        transient_flux = self.transient_face_flux_correction(
            previous_cell_velocity,
            previous_face_velocity,
            a_p,
        )
        flux = flux + transient_flux
        flux = self.spatial_face_velocity.enforce_boundary_flux(
            flux
        )
        free_divergence = self.pressure_equation.divergence_from_flux(
            flux
        )
        pressure_result = self.pressure_system.solve(
            -free_divergence,
            a_p,
            initial_pressure_state=pressure,
        )
        pressure_state = pressure_result.pressure
        pressure_flux = pressure_result.pressure_flux
        corrected_velocity = correct_cell_velocity(
            pressure_free_velocity,
            self.pressure_gradient.cell_gradient(pressure_state),
            self.discretization.cell_response(a_p),
        )
        corrected_flux = flux + pressure_flux
        corrected_flux = (
            self.spatial_face_velocity.enforce_boundary_flux(
                corrected_flux
            )
        )
        diagnostics = None
        if return_diagnostics:
            parts = pressure_result.flux_parts
            diagnostics = PisoPressureCorrectionDiagnostics(
                pressure_free_divergence_linf=linf_norm(
                    free_divergence
                ),
                pressure_corrected_divergence_linf=linf_norm(
                    self.pressure_equation.divergence_from_flux(
                        corrected_flux
                    )
                ),
                pressure_free_flux_linf=linf_norm(flux),
                pressure_corrected_flux_linf=linf_norm(corrected_flux),
                pressure_flux_linf=linf_norm(pressure_flux),
                pressure_orthogonal_flux_linf=linf_norm(
                    parts.orthogonal_flux
                ),
                pressure_cross_flux_linf=linf_norm(parts.cross_flux),
                pressure_boundary_flux_linf=linf_norm(
                    parts.boundary_pressure_flux
                ),
                transient_flux_correction_linf=linf_norm(
                    transient_flux
                ),
                pressure_nonorthogonal_iterations=(
                    pressure_result.nonorthogonal_iterations
                ),
            )
        return PisoPressureCorrectionStepResult(
            velocity=corrected_velocity,
            pressure=pressure_state,
            face_flux=corrected_flux,
            face_response_coefficient=(
                pressure_result.face_response_coefficient
            ),
            diagnostics=diagnostics,
        )

    def piso_pressure_corrector_loop(
        self,
        predicted_velocity: TensorLike,
        initial_pressure: TensorLike,
        a_p: TensorLike,
        momentum_matrix: CSRTensor,
        previous_cell_velocity: TensorLike,
        previous_face_velocity: TensorLike,
        *,
        step: int,
        time: float,
        corrector_diagnostics: list[PisoCorrectorDiagnostic],
        corrector_callback: PisoCorrectorCallback | None = None,
    ) -> tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
        """Run the repeated PISO pressure-corrector loop inside one time step.

        The first corrector uses the momentum-predicted velocity.  Later
        correctors first apply the neighbour-velocity compensation, then solve
        the same pressure-state correction step.
        """
        diagnostics_enabled = (
            self.controls.diagnostics_enabled or corrector_callback is not None
        )
        n_correctors = self.controls.n_correctors

        def record(
            correction: int,
            splitting_linf: float,
            current_velocity: TensorLike,
            current_pressure: TensorLike,
            result: PisoPressureCorrectionStepResult,
        ) -> None:
            diagnostics = result.diagnostics
            if diagnostics is None:
                return
            target_face_velocity = self.rhie_chow_face_velocity(
                result.velocity,
                result.pressure,
                result.face_response_coefficient,
            )
            target_face_velocity = (
                self.spatial_face_velocity.enforce_flux(
                    target_face_velocity,
                    self.spatial_face_velocity.compute_flux(
                        target_face_velocity
                    ),
                    result.face_flux,
                )
            )
            target_face_velocity = (
                self.spatial_face_velocity.enforce_boundary(
                    target_face_velocity
                )
            )
            target_flux_error = (
                self.spatial_face_velocity.compute_flux(
                    target_face_velocity
                )
                - result.face_flux
            )
            row = PisoCorrectorDiagnostic(
                step=step,
                time=time,
                corrector=correction,
                n_correctors=n_correctors,
                pressure_correction=diagnostics,
                operator_splitting_compensation_linf=splitting_linf,
                velocity_update_linf=linf_norm(
                    result.velocity - current_velocity
                ),
                pressure_update_linf=linf_norm(
                    result.pressure - current_pressure
                ),
                rhie_chow_flux_error_linf=linf_norm(
                    target_flux_error
                ),
            )
            corrector_diagnostics.append(row)
            if corrector_callback is not None:
                corrector_callback(row)

        velocity = predicted_velocity
        pressure = initial_pressure
        result = self.pressure_correction_step(
            velocity,
            pressure,
            a_p,
            previous_cell_velocity,
            previous_face_velocity,
            return_diagnostics=diagnostics_enabled,
        )
        record(1, 0.0, velocity, pressure, result)
        previous_corrector_velocity = velocity
        velocity = result.velocity
        pressure = result.pressure

        for correction in range(2, n_correctors + 1):
            intermediate_velocity = (
                self.piso_neighbour_velocity_correction(
                    velocity,
                    previous_corrector_velocity,
                    momentum_matrix,
                    a_p,
                )
            )
            splitting_linf = linf_norm(
                intermediate_velocity - velocity
            )
            previous_corrector_velocity = velocity
            result = self.pressure_correction_step(
                intermediate_velocity,
                pressure,
                a_p,
                previous_cell_velocity,
                previous_face_velocity,
                return_diagnostics=diagnostics_enabled,
            )
            record(
                correction,
                splitting_linf,
                velocity,
                pressure,
                result,
            )
            velocity = result.velocity
            pressure = result.pressure

        return (
            velocity,
            pressure,
            result.face_flux,
            result.face_response_coefficient,
        )

    def advance_time_step(
        self,
        previous_velocity: TensorLike,
        previous_face_velocity: TensorLike,
        previous_pressure: TensorLike,
        *,
        time: float,
        step: int,
        corrector_diagnostics: list[PisoCorrectorDiagnostic],
        corrector_callback: PisoCorrectorCallback | None = None,
    ) -> tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
        """Advance one time step through predictor, correctors, and face flux."""
        tau = self.controls.tau
        predictor = self.momentum.predict(
            previous_velocity,
            previous_face_velocity,
            previous_pressure,
            time + tau,
            pressure_gradient=(
                self.pressure_gradient.cell_gradient(previous_pressure)
            ),
        )
        predicted_velocity = predictor.cell_velocity
        a_p = predictor.response_diagonal
        momentum_matrix = predictor.momentum_matrix
        (
            current_velocity,
            current_pressure,
            face_flux,
            face_response_coefficient,
        ) = self.piso_pressure_corrector_loop(
            predicted_velocity,
            previous_pressure,
            a_p,
            momentum_matrix,
            previous_velocity,
            previous_face_velocity,
            step=step,
            time=time + tau,
            corrector_diagnostics=corrector_diagnostics,
            corrector_callback=corrector_callback,
        )
        next_face_velocity = self.rhie_chow_face_velocity(
            current_velocity,
            current_pressure,
            face_response_coefficient,
        )
        next_face_velocity = self.spatial_face_velocity.enforce_flux(
            next_face_velocity,
            self.spatial_face_velocity.compute_flux(next_face_velocity),
            face_flux,
        )
        next_face_velocity = self.spatial_face_velocity.enforce_boundary(
            next_face_velocity
        )
        return current_velocity, next_face_velocity, current_pressure, face_flux

    def solve(
        self,
        initial_velocity: TensorLike,
        initial_face_velocity: TensorLike,
        initial_pressure: TensorLike,
        *,
        snapshot_callback: PisoSnapshotCallback | None = None,
        corrector_callback: PisoCorrectorCallback | None = None,
    ) -> PisoSolveResult:
        """Advance all configured time steps from three explicit fields."""
        controls = self.controls
        discretization = self.discretization
        if initial_velocity.shape != (
            discretization.NC,
            discretization.GD,
        ):
            raise ValueError(
                "initial_velocity must have shape (NC, GD)."
            )
        if initial_face_velocity.shape != (
            discretization.NF,
            discretization.GD,
        ):
            raise ValueError(
                "initial_face_velocity must have shape (NF, GD)."
            )
        if initial_pressure.shape != (discretization.NC,):
            raise ValueError(
                "initial_pressure must have shape (NC,)."
            )
        current_velocity = initial_velocity
        current_face_velocity = initial_face_velocity
        current_pressure = initial_pressure
        current_face_flux = self.spatial_face_velocity.compute_flux(
            current_face_velocity
        )
        corrector_diagnostics: list[PisoCorrectorDiagnostic] = []
        for n in range(controls.time_steps):
            t = controls.duration[0] + n * controls.tau
            step = n + 1
            (
                current_velocity,
                current_face_velocity,
                current_pressure,
                current_face_flux,
            ) = self.advance_time_step(
                current_velocity,
                current_face_velocity,
                current_pressure,
                time=t,
                step=step,
                corrector_diagnostics=corrector_diagnostics,
                corrector_callback=corrector_callback,
            )
            if (
                snapshot_callback is not None
                and step >= controls.snapshot_start_step
                and (
                    step - controls.snapshot_start_step
                )
                % controls.snapshot_interval
                == 0
            ):
                snapshot = PisoSnapshot(
                    step=step,
                    time=t + controls.tau,
                    velocity=current_velocity,
                    face_velocity=current_face_velocity,
                    pressure=current_pressure,
                    face_flux=current_face_flux,
                )
                snapshot_callback(snapshot)
        return PisoSolveResult(
            velocity=current_velocity,
            pressure=current_pressure,
            face_velocity=current_face_velocity,
            face_flux=current_face_flux,
            corrector_diagnostics=tuple(corrector_diagnostics),
        )


__all__ = [
    "CollocatedPisoSolver",
    "PisoCorrectorCallback",
    "PisoSnapshotCallback",
    "PisoSourceFunction",
]

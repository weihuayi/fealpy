"""Collocated SIMPLE solver core for steady incompressible Navier-Stokes."""

from collections.abc import Callable
import logging

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .collocated_boundary_conditions import ResolvedSimpleBoundaryConditions
from .collocated_discretization import CollocatedDiscretization
from .collocated_momentum_equation import (
    CollocatedMomentumSpatialOperator,
    ComponentMomentumAlgebra,
    SteadyMomentumEquation,
)
from .collocated_pressure_system import (
    SimplePressureCorrectionSystem,
    resolve_pressure_system_closure,
)
from .collocated_pressure_equation import CollocatedPressureEquation
from .collocated_linear_solvers import CollocatedNSLinearSolvers
from .collocated_face_velocity_reconstruct import RhieChowInterpolation
from .collocated_spatial_face_velocity import CollocatedSpatialFaceVelocity
from .collocated_velocity_pressure_coupling import (
    correct_cell_velocity,
)
from .face_flux_reconstruct import FaceFluxReconstruct
from .simple_controls import (
    SimpleDiscretizationControls,
    SimpleIterationControls,
)
from .solver_controls import nonnegative_scalar, positive_scalar
from .simple_residual import (
    cell_l2_norm,
    collocated_mass_metrics,
)
from .simple_result import SimpleIterationResidual, SimpleSolveResult
from .solver_diagnostics import (
    inexact_inner_tolerance,
    log_simple_iteration,
    normalized_equation_residual,
)


class CollocatedSimpleSolver:
    """Algorithm core for steady collocated SIMPLE solves."""

    def __init__(
        self,
        *,
        diffusion_coef: float,
        convection_coef: float,
        source: Callable[[TensorLike], TensorLike],
        boundary_conditions: ResolvedSimpleBoundaryConditions,
        discretization_controls: SimpleDiscretizationControls,
        iteration_controls: SimpleIterationControls,
        linear_solvers: CollocatedNSLinearSolvers,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the reusable SIMPLE algorithm state."""
        if not isinstance(
            discretization_controls,
            SimpleDiscretizationControls,
        ):
            raise TypeError(
                "discretization_controls must be "
                "SimpleDiscretizationControls."
            )
        if not isinstance(iteration_controls, SimpleIterationControls):
            raise TypeError(
                "iteration_controls must be SimpleIterationControls."
            )
        if not isinstance(
            linear_solvers,
            CollocatedNSLinearSolvers,
        ):
            raise TypeError(
                "linear_solvers must be "
                "CollocatedNSLinearSolvers."
            )
        self.iteration_controls = iteration_controls
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
            logger.propagate = False
        self.logger = logger
        diffusion_coef = positive_scalar(diffusion_coef, "diffusion_coef")
        convection_coef = nonnegative_scalar(
            convection_coef,
            "convection_coef",
        )
        if not isinstance(
            boundary_conditions,
            ResolvedSimpleBoundaryConditions,
        ):
            raise TypeError(
                "CollocatedSimpleSolver requires "
                "ResolvedSimpleBoundaryConditions."
            )
        physical = boundary_conditions.physical
        discretization = CollocatedDiscretization(
            geometry=physical.geometry,
        )
        self.discretization = discretization
        self.pressure_gradient = physical.pressure_state.gradient
        self.pressure_correction_gradient = (
            boundary_conditions.pressure_correction.gradient
        )
        self.rhie_chow = RhieChowInterpolation(
            discretization.geometry,
            boundary_conditions.rhie_chow,
        )
        face_flux_reconstruct = FaceFluxReconstruct(
            geometry=discretization.geometry,
            method=discretization_controls.face_flux_correction_scheme,
            quadrature_order=(
                discretization_controls.face_flux_quadrature_order
            ),
            max_stencil_layers=(
                discretization_controls.face_flux_max_stencil_layers
            ),
            max_condition=(
                discretization_controls.face_flux_max_condition
            ),
        )
        self.spatial_face_velocity = CollocatedSpatialFaceVelocity(
            discretization=discretization,
            boundary=physical.velocity,
            scheme=(
                discretization_controls.spatial_face_velocity_scheme
            ),
            interpolation=(
                discretization_controls.rhie_chow_velocity_interpolation
            ),
            face_flux_reconstruct=face_flux_reconstruct,
        )
        momentum_spatial = CollocatedMomentumSpatialOperator(
            discretization=discretization,
            momentum_boundary=physical.momentum,
            diffusion_coef=diffusion_coef,
            convection_coef=convection_coef,
            diffusion_method=discretization_controls.diffusion_method,
            diffusion_nonorthogonal_eps=(
                discretization_controls.diffusion_nonorthogonal_eps
            ),
            face_interpolation=(
                discretization_controls.momentum_face_interpolation
            ),
        )
        momentum_algebra = ComponentMomentumAlgebra(
            discretization=discretization,
            linear_solver=linear_solvers.momentum,
        )
        self.momentum = SteadyMomentumEquation(
            spatial_operator=momentum_spatial,
            algebra=momentum_algebra,
            source=source,
            momentum_equation_relaxation=(
                iteration_controls.momentum_equation_relaxation
            ),
            momentum_nonorthogonal_max_iterations=(
                iteration_controls.momentum_nonorthogonal_max_iterations
            ),
            momentum_nonorthogonal_atol=(
                iteration_controls.momentum_nonorthogonal_atol
            ),
        )
        self.pressure_equation = CollocatedPressureEquation(
            discretization=discretization,
            diffusion_method=discretization_controls.diffusion_method,
            diffusion_nonorthogonal_eps=(
                discretization_controls.diffusion_nonorthogonal_eps
            ),
            response_interpolation=(
                discretization_controls.pressure_response_interpolation
            ),
        )
        pressure_boundary = boundary_conditions.pressure_correction
        pressure_closure = resolve_pressure_system_closure(
            pressure_boundary,
            self.pressure_equation,
            linear_solvers,
        )
        self.pressure_system = SimplePressureCorrectionSystem(
            pressure_equation=self.pressure_equation,
            boundary=pressure_boundary,
            closure=pressure_closure,
            pressure_nonorthogonal_max_iterations=(
                iteration_controls.pressure_nonorthogonal_max_iterations
            ),
            pressure_nonorthogonal_rtol=(
                iteration_controls.pressure_nonorthogonal_rtol
            ),
            pressure_nonorthogonal_atol=(
                iteration_controls.pressure_nonorthogonal_atol
            ),
        )

    def rhie_chow_face_velocity(
        self,
        u: TensorLike,
        p: TensorLike,
        response_coef: TensorLike,
        pressure_gradient: TensorLike,
    ) -> TensorLike:
        """Compose spatial reconstruction and Rhie-Chow stabilization."""
        base_face_velocity = self.spatial_face_velocity.reconstruct(u)
        face_velocity = self.rhie_chow.apply(
            base_face_velocity,
            p,
            response_coef,
            pressure_gradient,
        )
        return self.spatial_face_velocity.enforce_boundary(face_velocity)

    def solve(self) -> SimpleSolveResult:
        """Run the SIMPLE outer iteration."""
        controls = self.iteration_controls
        discretization = self.discretization
        geometry = discretization.geometry
        cell_measure = geometry.cell_measure
        NC = discretization.NC
        NF = discretization.NF
        GD = discretization.GD
        tol_momentum = controls.momentum_relative_tolerance
        tol_mass = controls.mass_relative_tolerance
        relax = controls.pressure_relaxation
        momentum_predictor_tol = inexact_inner_tolerance(
            controls.momentum_nonorthogonal_rtol,
            tol_momentum,
        )
        field_dtype = cell_measure.dtype
        field_device = bm.get_device(cell_measure)
        p = bm.zeros(NC, dtype=field_dtype, device=field_device)
        uf = bm.zeros(
            (NF, GD),
            dtype=field_dtype,
            device=field_device,
        )
        u = bm.zeros(
            (NC, GD),
            dtype=field_dtype,
            device=field_device,
        )
        pressure_gradient = self.pressure_gradient.cell_gradient(p)
        predictor = self.momentum.predict(
            p,
            uf,
            u,
            pressure_gradient=pressure_gradient,
            nonorthogonal_tolerance=momentum_predictor_tol,
        )
        correction_ap = predictor.correction_diagonal
        spatial_ap = predictor.spatial_diagonal
        u = predictor.cell_velocity
        residual_history = []
        termination_reason = "max_iterations"
        for iteration in range(1, controls.max_iterations + 1):
            spatial_response_coef = (
                self.pressure_equation.face_response_coefficient(
                    spatial_ap
                )
            )
            uf = self.rhie_chow_face_velocity(
                u,
                p,
                spatial_response_coef,
                pressure_gradient,
            )
            pressure_result = self.pressure_system.solve(
                uf,
                correction_ap,
            )
            p_corr = pressure_result.pressure_correction
            relaxed_p_corr = relax * p_corr
            relaxed_p_corr_gradient = (
                self.pressure_correction_gradient.cell_gradient(
                    relaxed_p_corr
                )
            )
            p += relaxed_p_corr
            u = correct_cell_velocity(
                u,
                relaxed_p_corr_gradient,
                discretization.cell_response(correction_ap),
            )
            pressure_gradient = self.pressure_gradient.cell_gradient(p)
            uf = self.rhie_chow_face_velocity(
                u,
                p,
                spatial_response_coef,
                pressure_gradient,
            )
            mass_metrics = collocated_mass_metrics(
                uf,
                geometry=geometry,
            )
            pressure_correction_l2 = cell_l2_norm(
                p_corr,
                geometry=geometry,
            )
            momentum = self.momentum.balance(
                p,
                u,
                uf,
                pressure_gradient=pressure_gradient,
            )
            momentum_norm = normalized_equation_residual(
                momentum.lhs,
                momentum.rhs,
                norm_weights=self.momentum.component_residual_weights,
                scale_mode="max",
            )
            residual = SimpleIterationResidual(
                iteration=iteration,
                mass_relative_l2=mass_metrics.relative_l2,
                mass_relative_l1=mass_metrics.relative_l1,
                mass_divergence_l2=mass_metrics.divergence_l2,
                mass_imbalance_linf=mass_metrics.absolute_linf,
                pressure_correction_l2=pressure_correction_l2,
                momentum_absolute_l2=momentum_norm.absolute,
                momentum_relative_l2=momentum_norm.relative,
                pressure_nonorthogonal_iterations=(
                    pressure_result.nonorthogonal_iterations
                ),
                momentum_nonorthogonal_iterations=(
                    predictor.nonorthogonal_iterations
                ),
                momentum_nonorthogonal_tolerance=momentum_predictor_tol,
            )
            residual_history.append(residual)
            log_simple_iteration(self.logger, iteration, residual)

            if (
                residual.momentum_relative_l2 < tol_momentum
                and residual.mass_relative_l2 < tol_mass
            ):
                termination_reason = "fixed_point_residuals"
                self.logger.info("Converged.")
                break
            if iteration == controls.max_iterations:
                break

            momentum_predictor_tol = inexact_inner_tolerance(
                controls.momentum_nonorthogonal_rtol,
                tol_momentum,
                residual.momentum_relative_l2,
            )
            predictor = self.momentum.predict(
                p,
                uf,
                u,
                pressure_gradient=pressure_gradient,
                nonorthogonal_tolerance=momentum_predictor_tol,
            )
            correction_ap = predictor.correction_diagonal
            spatial_ap = predictor.spatial_diagonal
            u = predictor.cell_velocity

        return SimpleSolveResult(
            velocity=u,
            pressure=p,
            face_velocity=uf,
            face_flux=self.spatial_face_velocity.compute_flux(uf),
            residual_history=tuple(residual_history),
            termination_reason=termination_reason,
        )

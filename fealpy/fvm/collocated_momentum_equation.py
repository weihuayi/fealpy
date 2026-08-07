"""Concrete collocated momentum operators shared by SIMPLE and PISO."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.fem import LinearForm
from fealpy.sparse import CSRTensor, spdiags
from fealpy.typing import TensorLike

from .collocated_boundary_conditions import ResolvedMomentumBoundary
from .collocated_discretization import CollocatedDiscretization
from .collocated_linear_solvers import LinearSystemSolver
from .collocated_spatial_face_velocity import (
    reconstruct_second_order_face_velocity,
)
from .convection_integrator import ConvectionMatrixAssembler
from .fvm_linear_solver import LinearSolveDiagnostics
from .scalar_cross_diffusion_integrator import (
    scalar_cross_diffusion_face_flux,
)
from .scalar_diffusion_integrator import ScalarDiffusionMatrixAssembler
from .scalar_source_integrator import ScalarSourceIntegrator
from .solver_diagnostics import (
    EquationResidual,
    equation_residual_converged,
    normalized_equation_residual,
    normalized_relaxed_equation_residual,
)

SteadyMomentumSource: TypeAlias = Callable[
    [TensorLike],
    TensorLike,
]
TransientMomentumSource: TypeAlias = Callable[
    [TensorLike, float],
    TensorLike,
]
MomentumResidualEvaluator: TypeAlias = Callable[
    [TensorLike, TensorLike, TensorLike],
    EquationResidual,
]


@dataclass(frozen=True)
class ComponentMomentumSystems:
    """One shared scalar matrix, component-major RHS, and scalar diagonals."""

    matrix: CSRTensor
    rhs: TensorLike
    relaxed_diagonal: TensorLike
    spatial_diagonal: TensorLike


@dataclass(frozen=True)
class ComponentMomentumSolveResult:
    """Component-major solution and one diagnostic per scalar solve."""

    velocity_dofs: TensorLike
    linear_solves: tuple[LinearSolveDiagnostics, ...]


@dataclass(frozen=True)
class NonorthogonalMomentumResult:
    """Call-local outcome of an explicit momentum diffusion correction."""

    velocity_dofs: TensorLike
    iterations: int
    residual: EquationResidual | None
    relative_update: float
    linear_solves: tuple[LinearSolveDiagnostics, ...]


@dataclass(frozen=True)
class MomentumPredictorResult:
    """Steady predictor with canonical scalar pressure responses."""

    cell_velocity: TensorLike
    correction_diagonal: TensorLike
    spatial_diagonal: TensorLike
    nonorthogonal_iterations: int
    nonorthogonal_tolerance: float
    nonorthogonal_residual: EquationResidual | None
    linear_solves: tuple[LinearSolveDiagnostics, ...]


@dataclass(frozen=True)
class TransientMomentumPredictorResult:
    """Transient predictor and the matrix required by PISO splitting."""

    cell_velocity: TensorLike
    response_diagonal: TensorLike
    momentum_matrix: CSRTensor
    nonorthogonal_iterations: int
    nonorthogonal_tolerance: float
    nonorthogonal_residual: EquationResidual | None
    linear_solves: tuple[LinearSolveDiagnostics, ...]


@dataclass(frozen=True)
class MomentumBalance:
    """Complete unrelaxed production momentum balance."""

    lhs: TensorLike
    rhs: TensorLike
    residual: TensorLike


@dataclass(frozen=True)
class MomentumTerms:
    """Named terms of the unrelaxed production momentum residual."""

    diffusion: TensorLike
    convection: TensorLike
    pressure: TensorLike
    source: TensorLike
    lhs: TensorLike
    rhs: TensorLike
    residual: TensorLike


class CollocatedMomentumSpatialOperator:
    """Own the spatial momentum discretization and its immutable templates."""

    def __init__(
        self,
        *,
        discretization: CollocatedDiscretization,
        momentum_boundary: ResolvedMomentumBoundary,
        diffusion_coef: float,
        convection_coef: float,
        diffusion_method: str,
        diffusion_nonorthogonal_eps: float,
        face_interpolation: str,
    ) -> None:
        self.discretization = discretization
        self.momentum_boundary = momentum_boundary
        self.diffusion_coef = diffusion_coef
        self.convection_coef = convection_coef
        self.diffusion_method = diffusion_method
        self.diffusion_nonorthogonal_eps = diffusion_nonorthogonal_eps
        self.face_interpolation = face_interpolation
        self._diffusion_matrix_template = None
        self._convection_assembler = None

    @property
    def diffusion_matrix_template(self) -> CSRTensor:
        """Return the lazily assembled, read-only scalar diffusion template."""
        matrix = self._diffusion_matrix_template
        if matrix is None:
            matrix = ScalarDiffusionMatrixAssembler(
                self.discretization.space,
                geometry=self.discretization.geometry,
                method=self.diffusion_method,
                nonorthogonal_eps=self.diffusion_nonorthogonal_eps,
            ).assembly(self.diffusion_coef)
            self._diffusion_matrix_template = matrix
        return matrix

    @property
    def convection_assembler(self) -> ConvectionMatrixAssembler:
        """Return the one assembler bound to the selected face interpolation."""
        assembler = self._convection_assembler
        if assembler is None:
            assembler = ConvectionMatrixAssembler(
                self.discretization.space,
                interpolation=self.face_interpolation,
                geometry=self.discretization.geometry,
            )
            self._convection_assembler = assembler
        return assembler

    def working_matrix(
        self,
        face_velocity: TensorLike,
    ) -> CSRTensor:
        """Build this iteration's scalar spatial matrix from fixed templates."""
        matrix = self.diffusion_matrix_template
        if self.convection_coef != 0.0:
            matrix = matrix + self.convection_assembler.assembly(
                self.convection_coef * face_velocity
            )
        return matrix

    def source_vector(
        self,
        source: SteadyMomentumSource,
    ) -> TensorLike:
        """Assemble one component-major, cell-integrated physical source."""
        return LinearForm(
            self.discretization.velocity_space
        ).add_integrator(
            ScalarSourceIntegrator(
                source,
                q=self.discretization.degree + 2,
                geometry=self.discretization.geometry,
            )
        ).assembly()

    def second_order_face_velocity(
        self,
        cell_velocity: TensorLike,
    ) -> TensorLike:
        """Blend owner/neighbour linear reconstructions at face centres."""
        gradient = self.momentum_boundary.velocity.gradient.cell_gradient(
            cell_velocity
        )
        return reconstruct_second_order_face_velocity(
            self.discretization.geometry,
            gradient,
            cell_velocity,
        )

    def apply_boundary(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
        *,
        face_velocity: TensorLike,
        cell_velocity: TensorLike,
    ) -> tuple[CSRTensor, TensorLike]:
        """Apply the already resolved momentum boundary operators."""
        momentum = self.momentum_boundary
        matrix, rhs = momentum.apply_diffusion(
            matrix,
            rhs,
            self.diffusion_coef,
        )
        if momentum.traction.faces.shape[0] > 0:
            rhs = rhs + self.discretization.cell_vector_to_dofs(
                momentum.source(rhs)
            )
        if momentum.velocity.neumann_faces.shape[0] > 0:
            rhs = rhs + self.discretization.cell_vector_to_dofs(
                momentum.diffusion_source(self.diffusion_coef, rhs)
            )
        if self.convection_coef != 0.0:
            convection_face_velocity = (
                self.convection_coef * face_velocity
            )
            if momentum.velocity.natural_faces.shape[0] > 0:
                natural_diagonal = momentum.natural_convection_diagonal(
                    convection_face_velocity,
                )
                matrix = matrix + spdiags(
                    natural_diagonal,
                    0,
                    matrix.shape[0],
                    matrix.shape[1],
                    index_dtype=matrix.itype,
                )
            if (
                momentum.velocity.dirichlet_operator.faces.shape[0] > 0
            ):
                rhs = (
                    momentum.velocity.dirichlet_operator.apply_convection(
                        rhs,
                        convection_face_velocity,
                        components=self.discretization.GD,
                    )
                )
            if momentum.traction.faces.shape[0] > 0:
                convection_source = momentum.convection_source(
                    convection_face_velocity,
                    cell_velocity,
                    self.second_order_face_velocity(cell_velocity),
                )
                rhs = rhs + self.discretization.cell_vector_to_dofs(
                    convection_source
                )
        return matrix, rhs

    def pressure_source(
        self,
        pressure: TensorLike,
        *,
        pressure_gradient: TensorLike,
    ) -> TensorLike:
        """Return the resolved cell-integrated pressure force."""
        cell_force = (
            self.discretization.geometry.cell_measure[:, None]
            * pressure_gradient
        )
        cell_force = self.momentum_boundary.apply_pressure_force(
            cell_force,
            pressure,
            pressure_gradient,
        )
        return self.discretization.cell_vector_to_dofs(cell_force)

    def boundary_source(self, reference: TensorLike) -> TensorLike:
        """Return only the prescribed traction source in algebraic layout."""
        return self.discretization.cell_vector_to_dofs(
            self.momentum_boundary.source(reference)
        )

    def boundary_corrected_face_gradient(
        self,
        velocity: TensorLike,
        *,
        interpolation_method: str,
    ) -> TensorLike:
        """Return face velocity gradients with the resolved boundary data."""
        velocity_boundary = self.momentum_boundary.velocity
        cell_gradient = velocity_boundary.gradient.cell_gradient(velocity)
        return velocity_boundary.correct_face_gradient(
            cell_gradient,
            velocity,
            interpolation_method=interpolation_method,
        )

    def nonorthogonal_rhs(self, velocity: TensorLike) -> TensorLike:
        """Assemble explicit cross-diffusion in component-major layout."""
        grad_f = self.boundary_corrected_face_gradient(
            velocity,
            interpolation_method="average",
        )
        geometry = self.discretization.geometry
        decomposition = geometry.diffusion_face_decomposition(
            self.diffusion_method,
            eps=self.diffusion_nonorthogonal_eps,
        )
        correction_vector = (
            bm.zeros_like(geometry.S_f)
            if self.diffusion_method == "uncorrected"
            else decomposition.T_f
        )
        face_flux = scalar_cross_diffusion_face_flux(
            self.discretization.velocity_space,
            geometry,
            geometry.face_to_cell,
            grad_f=grad_f,
            coef=self.diffusion_coef,
            correction_vector=correction_vector,
            boundary_policy="all",
        )
        face_flux = self.momentum_boundary.apply_cross_diffusion_flux(
            face_flux
        )
        return self.discretization.cell_vector_to_dofs(
            geometry.scatter_face_flux_to_cells(face_flux)
        )


class ComponentMomentumAlgebra:
    """Component-major algebra shared by concrete momentum equations."""

    def __init__(
        self,
        *,
        discretization: CollocatedDiscretization,
        linear_solver: LinearSystemSolver,
    ) -> None:
        self.discretization = discretization
        self.linear_solver = linear_solver

    def component_systems(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
        previous_velocity: TensorLike,
        *,
        relaxation: float,
    ) -> ComponentMomentumSystems:
        """Build the single shared scalar operator for all components."""
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(
                "momentum equation relaxation alpha must be in (0, 1]."
            )
        discretization = self.discretization
        spatial_diagonal = matrix.diags().values
        working_matrix = matrix
        working_rhs = rhs
        if relaxation < 1.0:
            delta = (1.0 / relaxation - 1.0) * spatial_diagonal
            working_matrix = matrix + spdiags(
                delta,
                0,
                matrix.shape[0],
                matrix.shape[1],
                index_dtype=matrix.itype,
            )
            previous_dofs = discretization.cell_vector_to_dofs(
                previous_velocity
            )
            nc = discretization.NC
            working_rhs = bm.concatenate(
                [
                    rhs[c * nc : (c + 1) * nc]
                    + delta * previous_dofs[c * nc : (c + 1) * nc]
                    for c in range(discretization.GD)
                ],
                axis=0,
            )
        return ComponentMomentumSystems(
            matrix=working_matrix,
            rhs=working_rhs,
            relaxed_diagonal=working_matrix.diags().values,
            spatial_diagonal=spatial_diagonal,
        )

    def solve_components(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
    ) -> ComponentMomentumSolveResult:
        """Solve each scalar velocity component in component-major order."""
        discretization = self.discretization
        nc = discretization.NC
        results = tuple(
            self.linear_solver.solve(
                matrix,
                rhs[c * nc : (c + 1) * nc],
            )
            for c in range(discretization.GD)
        )
        return ComponentMomentumSolveResult(
            velocity_dofs=bm.concatenate(
                [result.solution for result in results],
                axis=0,
            ),
            linear_solves=tuple(
                result.diagnostics for result in results
            ),
        )

    def matrix_action(
        self,
        matrix: CSRTensor,
        dofs: TensorLike,
    ) -> TensorLike:
        """Apply shared scalar component matrices to velocity dofs."""
        discretization = self.discretization
        nc = discretization.NC
        return bm.concatenate(
            [
                matrix @ dofs[c * nc : (c + 1) * nc]
                for c in range(discretization.GD)
            ],
            axis=0,
        )


def iterate_nonorthogonal_momentum(
    *,
    spatial_operator: CollocatedMomentumSpatialOperator,
    algebra: ComponentMomentumAlgebra,
    matrix: CSRTensor,
    base_rhs: TensorLike,
    initial_velocity_dofs: TensorLike,
    initial_linear_solves: tuple[LinearSolveDiagnostics, ...],
    residual_evaluator: MomentumResidualEvaluator,
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> NonorthogonalMomentumResult:
    """Perform call-local Picard correction for cross diffusion."""
    if max_iterations == 0:
        return NonorthogonalMomentumResult(
            velocity_dofs=initial_velocity_dofs,
            iterations=0,
            residual=None,
            relative_update=0.0,
            linear_solves=initial_linear_solves,
        )

    corrected = initial_velocity_dofs
    linear_solves = list(initial_linear_solves)
    relative_update = 0.0
    residual = None
    for correction in range(max_iterations + 1):
        cell_velocity = algebra.discretization.dofs_to_cell_vector(
            corrected
        )
        corrected_rhs = (
            base_rhs + spatial_operator.nonorthogonal_rhs(cell_velocity)
        )
        lhs = algebra.matrix_action(matrix, corrected)
        residual = residual_evaluator(
            lhs,
            corrected_rhs,
            corrected,
        )
        if equation_residual_converged(
            residual,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            return NonorthogonalMomentumResult(
                velocity_dofs=corrected,
                iterations=correction,
                residual=residual,
                relative_update=relative_update,
                linear_solves=tuple(linear_solves),
            )
        if correction == max_iterations:
            break
        solve_result = algebra.solve_components(
            matrix,
            corrected_rhs,
        )
        next_velocity = solve_result.velocity_dofs
        linear_solves.extend(solve_result.linear_solves)
        update = float(
            bm.to_numpy(bm.linalg.norm(next_velocity - corrected))
        )
        scale = float(bm.to_numpy(bm.linalg.norm(next_velocity)))
        relative_update = update / max(scale, 1.0e-30)
        corrected = next_velocity

    raise RuntimeError(
        "momentum non-orthogonal correction did not converge before "
        "the configured maximum iterations"
    )


class SteadyMomentumEquation:
    """Compose the spatial operator, steady source, and SIMPLE relaxation."""

    def __init__(
        self,
        *,
        spatial_operator: CollocatedMomentumSpatialOperator,
        algebra: ComponentMomentumAlgebra,
        source: SteadyMomentumSource,
        momentum_equation_relaxation: float,
        momentum_nonorthogonal_max_iterations: int,
        momentum_nonorthogonal_atol: float,
    ) -> None:
        self.algebra = algebra
        self.source = source
        self.momentum_equation_relaxation = float(
            momentum_equation_relaxation
        )
        self.momentum_nonorthogonal_max_iterations = int(
            momentum_nonorthogonal_max_iterations
        )
        self.momentum_nonorthogonal_atol = float(
            momentum_nonorthogonal_atol
        )
        self.spatial_operator = spatial_operator
        self._source_template = None
        geometry = spatial_operator.discretization.geometry
        self.component_residual_weights = 1.0 / bm.tile(
            geometry.cell_measure,
            (spatial_operator.discretization.GD,),
        )

    @property
    def source_template(self) -> TensorLike:
        """Return the lazily assembled, read-only steady source template."""
        source = self._source_template
        if source is None:
            source = self.spatial_operator.source_vector(self.source)
            self._source_template = source
        return source

    def working_system(
        self,
        pressure: TensorLike,
        face_velocity: TensorLike,
        cell_velocity: TensorLike,
        *,
        pressure_gradient: TensorLike,
        relaxation: float,
    ) -> ComponentMomentumSystems:
        matrix = self.spatial_operator.working_matrix(face_velocity)
        rhs = bm.copy(self.source_template)
        matrix, rhs = self.spatial_operator.apply_boundary(
            matrix,
            rhs,
            face_velocity=face_velocity,
            cell_velocity=cell_velocity,
        )
        rhs = rhs - self.spatial_operator.pressure_source(
            pressure,
            pressure_gradient=pressure_gradient,
        )
        return self.algebra.component_systems(
            matrix,
            rhs,
            cell_velocity,
            relaxation=relaxation,
        )

    def predict(
        self,
        pressure: TensorLike,
        face_velocity: TensorLike,
        previous_velocity: TensorLike,
        *,
        pressure_gradient: TensorLike,
        nonorthogonal_tolerance: float,
    ) -> MomentumPredictorResult:
        """Solve one steady momentum predictor."""
        systems = self.working_system(
            pressure,
            face_velocity,
            previous_velocity,
            pressure_gradient=pressure_gradient,
            relaxation=self.momentum_equation_relaxation,
        )
        linear_result = self.algebra.solve_components(
            systems.matrix,
            systems.rhs,
        )
        discretization = self.spatial_operator.discretization
        weights = self.component_residual_weights
        previous_velocity_dofs = discretization.cell_vector_to_dofs(
            previous_velocity
        )
        component_relaxation_diagonal = (
            discretization.component_cell_diagonal(
                systems.relaxed_diagonal - systems.spatial_diagonal
            )
        )

        def residual_evaluator(
            lhs: TensorLike,
            corrected_rhs: TensorLike,
            corrected: TensorLike,
        ) -> EquationResidual:
            return normalized_relaxed_equation_residual(
                lhs,
                corrected_rhs,
                corrected,
                previous_velocity_dofs,
                component_relaxation_diagonal,
                norm_weights=weights,
            )

        correction = iterate_nonorthogonal_momentum(
            spatial_operator=self.spatial_operator,
            algebra=self.algebra,
            matrix=systems.matrix,
            base_rhs=systems.rhs,
            initial_velocity_dofs=linear_result.velocity_dofs,
            initial_linear_solves=linear_result.linear_solves,
            residual_evaluator=residual_evaluator,
            max_iterations=(
                self.momentum_nonorthogonal_max_iterations
            ),
            relative_tolerance=nonorthogonal_tolerance,
            absolute_tolerance=self.momentum_nonorthogonal_atol,
        )
        return MomentumPredictorResult(
            cell_velocity=discretization.dofs_to_cell_vector(
                correction.velocity_dofs
            ),
            correction_diagonal=systems.relaxed_diagonal,
            spatial_diagonal=systems.spatial_diagonal,
            nonorthogonal_iterations=correction.iterations,
            nonorthogonal_tolerance=nonorthogonal_tolerance,
            nonorthogonal_residual=correction.residual,
            linear_solves=correction.linear_solves,
        )

    def balance(
        self,
        pressure: TensorLike,
        velocity: TensorLike,
        face_velocity: TensorLike,
        *,
        pressure_gradient: TensorLike,
    ) -> MomentumBalance:
        """Return the unrelaxed production momentum balance."""
        systems = self.working_system(
            pressure,
            face_velocity,
            velocity,
            pressure_gradient=pressure_gradient,
            relaxation=1.0,
        )
        discretization = self.spatial_operator.discretization
        velocity_dofs = discretization.cell_vector_to_dofs(velocity)
        rhs = systems.rhs
        if (
            self.momentum_nonorthogonal_max_iterations > 0
        ):
            rhs = rhs + self.spatial_operator.nonorthogonal_rhs(velocity)
        lhs = self.algebra.matrix_action(
            systems.matrix,
            velocity_dofs,
        )
        return MomentumBalance(
            lhs=lhs,
            rhs=rhs,
            residual=lhs - rhs,
        )

    def terms(
        self,
        pressure: TensorLike,
        velocity: TensorLike,
        face_velocity: TensorLike,
        *,
        pressure_gradient: TensorLike,
    ) -> MomentumTerms:
        """Return named terms of the unrelaxed production residual."""
        balance = self.balance(
            pressure,
            velocity,
            face_velocity,
            pressure_gradient=pressure_gradient,
        )
        discretization = self.spatial_operator.discretization
        nc = discretization.NC
        gd = discretization.GD
        velocity_dofs = discretization.cell_vector_to_dofs(velocity)
        zero_rhs = bm.zeros_like(velocity_dofs)
        boundary = self.spatial_operator.momentum_boundary
        diffusion_boundary_source = (
            discretization.cell_vector_to_dofs(
                boundary.diffusion_source(
                    self.spatial_operator.diffusion_coef,
                    velocity_dofs,
                )
            )
        )
        diffusion_matrix, diffusion_rhs = boundary.apply_diffusion(
            self.spatial_operator.diffusion_matrix_template,
            zero_rhs,
            self.spatial_operator.diffusion_coef,
        )
        diffusion_systems = self.algebra.component_systems(
            diffusion_matrix,
            diffusion_rhs,
            velocity,
            relaxation=1.0,
        )
        diffusion = (
            self.algebra.matrix_action(
                diffusion_systems.matrix,
                velocity_dofs,
            )
            - diffusion_systems.rhs
            - diffusion_boundary_source
        )
        if (
            self.momentum_nonorthogonal_max_iterations > 0
        ):
            diffusion = (
                diffusion
                - self.spatial_operator.nonorthogonal_rhs(velocity)
            )
        pressure_term = self.spatial_operator.pressure_source(
            pressure,
            pressure_gradient=pressure_gradient,
        )
        source_term = (
            self.source_template
            + self.spatial_operator.boundary_source(
                velocity_dofs
            )
        )
        convection = (
            balance.residual
            - diffusion
            - pressure_term
            + source_term
        )
        return MomentumTerms(
            diffusion=diffusion,
            convection=convection,
            pressure=pressure_term,
            source=source_term,
            lhs=balance.lhs,
            rhs=balance.rhs,
            residual=balance.residual,
        )


class TransientMomentumEquation:
    """Compose the shared spatial operator with backward-Euler time terms."""

    def __init__(
        self,
        *,
        spatial_operator: CollocatedMomentumSpatialOperator,
        algebra: ComponentMomentumAlgebra,
        source: TransientMomentumSource,
        density: float,
        time_step: float,
        momentum_nonorthogonal_max_iterations: int,
        momentum_nonorthogonal_rtol: float,
        momentum_nonorthogonal_atol: float,
    ) -> None:
        self.algebra = algebra
        self.source = source
        self.density = density
        self.time_step = time_step
        self.momentum_nonorthogonal_max_iterations = int(
            momentum_nonorthogonal_max_iterations
        )
        self.momentum_nonorthogonal_rtol = float(
            momentum_nonorthogonal_rtol
        )
        self.momentum_nonorthogonal_atol = float(
            momentum_nonorthogonal_atol
        )
        self.spatial_operator = spatial_operator
        geometry = spatial_operator.discretization.geometry
        self.component_residual_weights = 1.0 / bm.tile(
            geometry.cell_measure,
            (spatial_operator.discretization.GD,),
        )
        self._time_diagonal = (
            self.density * geometry.cell_measure / self.time_step
        )
        nc = spatial_operator.discretization.NC
        self._time_matrix = CSRTensor(
            crow=bm.arange(
                nc + 1,
                dtype=bm.int64,
                device=bm.get_device(self._time_diagonal),
            ),
            col=bm.arange(
                nc,
                dtype=bm.int64,
                device=bm.get_device(self._time_diagonal),
            ),
            values=self._time_diagonal,
            spshape=(nc, nc),
        )

    def time_diagonal(self) -> TensorLike:
        """Return ``rho*V/dt`` as one canonical cell scalar."""
        return self._time_diagonal

    def time_matrix(self) -> CSRTensor:
        """Return the scalar backward-Euler cell mass matrix."""
        return self._time_matrix

    def time_source(
        self,
        previous_velocity: TensorLike,
    ) -> TensorLike:
        """Return the backward-Euler old-time source."""
        diagonal = self.time_diagonal()
        return self.spatial_operator.discretization.cell_vector_to_dofs(
            previous_velocity * diagonal[:, None]
        )

    def matrix_action(
        self,
        matrix: CSRTensor,
        velocity_dofs: TensorLike,
    ) -> TensorLike:
        """Apply the transient component matrices to velocity dofs."""
        return self.algebra.matrix_action(matrix, velocity_dofs)

    def predict(
        self,
        previous_velocity: TensorLike,
        previous_face_velocity: TensorLike,
        pressure: TensorLike,
        time: float,
        *,
        pressure_gradient: TensorLike,
    ) -> TransientMomentumPredictorResult:
        """Solve one backward-Euler transient momentum predictor."""
        matrix = self.spatial_operator.working_matrix(
            previous_face_velocity
        )
        matrix = matrix + self.time_matrix()

        @cartesian
        def source(points: TensorLike) -> TensorLike:
            return self.source(points, time)

        rhs = self.spatial_operator.source_vector(source)
        rhs = rhs + self.time_source(previous_velocity)
        matrix, rhs = self.spatial_operator.apply_boundary(
            matrix,
            rhs,
            face_velocity=previous_face_velocity,
            cell_velocity=previous_velocity,
        )
        rhs = rhs - self.spatial_operator.pressure_source(
            pressure,
            pressure_gradient=pressure_gradient,
        )
        systems = self.algebra.component_systems(
            matrix,
            rhs,
            previous_velocity,
            relaxation=1.0,
        )
        linear_result = self.algebra.solve_components(
            systems.matrix,
            systems.rhs,
        )
        discretization = self.spatial_operator.discretization
        weights = self.component_residual_weights

        def residual_evaluator(
            lhs: TensorLike,
            corrected_rhs: TensorLike,
            _corrected: TensorLike,
        ) -> EquationResidual:
            return normalized_equation_residual(
                lhs,
                corrected_rhs,
                norm_weights=weights,
            )

        correction = iterate_nonorthogonal_momentum(
            spatial_operator=self.spatial_operator,
            algebra=self.algebra,
            matrix=systems.matrix,
            base_rhs=systems.rhs,
            initial_velocity_dofs=linear_result.velocity_dofs,
            initial_linear_solves=linear_result.linear_solves,
            residual_evaluator=residual_evaluator,
            max_iterations=self.momentum_nonorthogonal_max_iterations,
            relative_tolerance=self.momentum_nonorthogonal_rtol,
            absolute_tolerance=self.momentum_nonorthogonal_atol,
        )
        return TransientMomentumPredictorResult(
            cell_velocity=(
                self.spatial_operator.discretization.dofs_to_cell_vector(
                    correction.velocity_dofs
                )
            ),
            response_diagonal=systems.relaxed_diagonal,
            momentum_matrix=systems.matrix,
            nonorthogonal_iterations=correction.iterations,
            nonorthogonal_tolerance=(
                self.momentum_nonorthogonal_rtol
            ),
            nonorthogonal_residual=correction.residual,
            linear_solves=correction.linear_solves,
        )


__all__ = [
    "CollocatedMomentumSpatialOperator",
    "ComponentMomentumAlgebra",
    "ComponentMomentumSolveResult",
    "ComponentMomentumSystems",
    "MomentumBalance",
    "MomentumPredictorResult",
    "MomentumTerms",
    "NonorthogonalMomentumResult",
    "SteadyMomentumEquation",
    "TransientMomentumEquation",
    "TransientMomentumPredictorResult",
    "iterate_nonorthogonal_momentum",
]

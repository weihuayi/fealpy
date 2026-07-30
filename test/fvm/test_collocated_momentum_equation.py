import numpy as np


def _solver():
    from test.fvm.test_collocated_simple_solver_structure import _cavity_solver

    return _cavity_solver(convection_coef=0.0)


def test_simple_owns_steady_momentum_operator_with_scalar_response():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm.collocated_momentum_equation import (
        MomentumPredictorResult,
        SteadyMomentumEquation,
    )

    solver = _solver()
    assert isinstance(solver.momentum, SteadyMomentumEquation)

    pressure = bm.zeros(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    face_velocity = bm.zeros((solver.discretization.NF, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    velocity = bm.zeros((solver.discretization.NC, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    result = solver.momentum.predict(
        pressure,
        face_velocity,
        velocity,
        pressure_gradient=solver.pressure_gradient.cell_gradient(pressure),
        nonorthogonal_tolerance=(
            solver.iteration_controls.momentum_nonorthogonal_rtol
        ),
    )

    assert isinstance(result, MomentumPredictorResult)
    assert result.cell_velocity.shape == (solver.discretization.NC, solver.discretization.GD)
    assert result.correction_diagonal.shape == (solver.discretization.NC,)
    assert result.spatial_diagonal.shape == (solver.discretization.NC,)
    assert result.nonorthogonal_iterations >= 0


def test_component_momentum_algebra_has_no_spatial_dependency():
    import inspect
    from fealpy.fvm.collocated_momentum_equation import (
        ComponentMomentumAlgebra,
    )

    parameters = inspect.signature(
        ComponentMomentumAlgebra.__init__
    ).parameters
    assert "discretization" in parameters
    assert "spatial_operator" not in parameters
    assert not hasattr(
        ComponentMomentumAlgebra,
        "iterate_nonorthogonal",
    )
    assert not hasattr(
        ComponentMomentumAlgebra,
        "correct_relaxed_nonorthogonal",
    )
    assert not hasattr(
        ComponentMomentumAlgebra,
        "correct_unrelaxed_nonorthogonal",
    )


def test_steady_momentum_uses_only_call_local_nonorthogonal_rtol():
    from fealpy.fvm.collocated_momentum_equation import SteadyMomentumEquation

    solver = _solver()
    current = solver.momentum

    equation = SteadyMomentumEquation(
        spatial_operator=current.spatial_operator,
        algebra=current.algebra,
        source=current.source,
        momentum_equation_relaxation=(
            current.momentum_equation_relaxation
        ),
        momentum_nonorthogonal_max_iterations=(
            current.momentum_nonorthogonal_max_iterations
        ),
        momentum_nonorthogonal_atol=current.momentum_nonorthogonal_atol,
    )

    assert isinstance(equation, SteadyMomentumEquation)


def test_momentum_spatial_operator_accepts_only_required_boundaries():
    from fealpy.fvm.collocated_momentum_equation import (
        CollocatedMomentumSpatialOperator,
    )

    solver = _solver()
    current = solver.momentum.spatial_operator

    spatial_operator = CollocatedMomentumSpatialOperator(
        discretization=current.discretization,
        momentum_boundary=current.momentum_boundary,
        diffusion_coef=current.diffusion_coef,
        convection_coef=current.convection_coef,
        diffusion_method=current.diffusion_method,
        diffusion_nonorthogonal_eps=current.diffusion_nonorthogonal_eps,
        face_interpolation=current.face_interpolation,
    )

    assert isinstance(spatial_operator, CollocatedMomentumSpatialOperator)


def test_momentum_system_uses_one_shared_scalar_matrix():
    from fealpy.backend import backend_manager as bm

    solver = _solver()
    pressure = bm.zeros(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    face_velocity = bm.zeros(
        (solver.discretization.NF, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    velocity = bm.zeros(
        (solver.discretization.NC, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    systems = solver.momentum.working_system(
        pressure,
        face_velocity,
        velocity,
        pressure_gradient=solver.pressure_gradient.cell_gradient(pressure),
        relaxation=1.0,
    )

    assert systems.matrix.shape == (solver.discretization.NC, solver.discretization.NC)


def test_steady_momentum_templates_are_owned_and_not_mutated():
    from fealpy.backend import backend_manager as bm

    solver = _solver()
    equation = solver.momentum
    spatial = equation.spatial_operator
    source_before = bm.copy(equation.source_template)
    matrix_before = bm.copy(
        spatial.diffusion_matrix_template.values
    )

    pressure = bm.zeros(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    face_velocity = bm.zeros((solver.discretization.NF, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    velocity = bm.zeros((solver.discretization.NC, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    pressure_gradient = solver.pressure_gradient.cell_gradient(pressure)
    tolerance = solver.iteration_controls.momentum_nonorthogonal_rtol
    equation.predict(
        pressure,
        face_velocity,
        velocity,
        pressure_gradient=pressure_gradient,
        nonorthogonal_tolerance=tolerance,
    )
    equation.predict(
        pressure,
        face_velocity,
        velocity,
        pressure_gradient=pressure_gradient,
        nonorthogonal_tolerance=tolerance,
    )

    np.testing.assert_allclose(
        bm.to_numpy(equation.source_template),
        bm.to_numpy(source_before),
    )
    np.testing.assert_allclose(
        bm.to_numpy(spatial.diffusion_matrix_template.values),
        bm.to_numpy(matrix_before),
    )


def test_empty_traction_does_not_reconstruct_second_order_face_velocity(
    monkeypatch,
):
    from fealpy.backend import backend_manager as bm
    from test.fvm.test_collocated_simple_solver_structure import (
        _cavity_solver,
    )

    solver = _cavity_solver(convection_coef=1.0)
    spatial = solver.momentum.spatial_operator
    assert spatial.momentum_boundary.traction.faces.shape[0] == 0

    def reject_reconstruction(_cell_velocity):
        raise AssertionError("empty traction must not reconstruct face velocity")

    monkeypatch.setattr(
        spatial,
        "second_order_face_velocity",
        reject_reconstruction,
    )
    pressure = bm.zeros(
        solver.discretization.NC,
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    face_velocity = bm.zeros(
        (solver.discretization.NF, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    cell_velocity = bm.zeros(
        (solver.discretization.NC, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )

    solver.momentum.working_system(
        pressure,
        face_velocity,
        cell_velocity,
        pressure_gradient=solver.pressure_gradient.cell_gradient(pressure),
        relaxation=1.0,
    )


def test_momentum_residual_weights_are_fixed_session_cache():
    steady = _solver().momentum
    transient = __import__(
        "test.fvm.test_collocated_piso_solver_structure",
        fromlist=["_piso_model"],
    )._piso_model().solver.momentum

    assert (
        steady.component_residual_weights
        is steady.component_residual_weights
    )
    assert (
        transient.component_residual_weights
        is transient.component_residual_weights
    )


def test_transient_time_terms_are_fixed_session_cache():
    momentum = __import__(
        "test.fvm.test_collocated_piso_solver_structure",
        fromlist=["_piso_model"],
    )._piso_model().solver.momentum

    assert momentum.time_diagonal() is momentum.time_diagonal()
    assert momentum.time_matrix() is momentum.time_matrix()
def test_component_momentum_solve_returns_each_linear_diagnostic():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm.collocated_momentum_equation import (
        ComponentMomentumAlgebra,
        ComponentMomentumSolveResult,
    )
    from fealpy.fvm.fvm_linear_solver import (
        LinearSolveDiagnostics,
        LinearSolveResult,
    )

    class Discretization:
        NC = 2
        GD = 2

    class LinearSolver:
        def __init__(self):
            self.calls = 0

        def solve(self, matrix, rhs):
            self.calls += 1
            return LinearSolveResult(
                solution=rhs + self.calls,
                diagnostics=LinearSolveDiagnostics(
                    provider="test",
                    solver=f"component-{self.calls}",
                    iterations=self.calls,
                    converged=True,
                    provider_code=0,
                    relative_residual=0.0,
                ),
            )

    algebra = ComponentMomentumAlgebra(
        discretization=Discretization(),
        linear_solver=LinearSolver(),
    )
    result = algebra.solve_components(
        "matrix",
        bm.array([1.0, 2.0, 3.0, 4.0]),
    )

    assert isinstance(result, ComponentMomentumSolveResult)
    assert bm.to_numpy(result.velocity_dofs).tolist() == [
        2.0,
        3.0,
        5.0,
        6.0,
    ]
    assert tuple(
        diagnostic.solver for diagnostic in result.linear_solves
    ) == ("component-1", "component-2")

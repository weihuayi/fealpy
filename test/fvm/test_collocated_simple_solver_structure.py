import numpy as np
import pytest


def _cavity_solver(
    convection_coef=None,
    *,
    discretization_controls=None,
    iteration_controls=None,
    pressure_system_controls=None,
    linear_solvers=None,
    resolution=2,
):
    from fealpy.fvm import (
        CollocatedPressureSystemControls,
        CollocatedSimpleSolver,
        LidDrivenCavityCase,
        PDEBoundaryConditions,
        PressureClosureKind,
        SimpleDiscretizationControls,
        SimpleIterationControls,
        build_collocated_ns_linear_solvers,
        resolve_simple_boundary_conditions,
    )

    case = LidDrivenCavityCase(re=10.0)
    mesh = case.init_mesh["uniform_quad"](nx=resolution, ny=resolution)
    if linear_solvers is None:
        linear_solvers = build_collocated_ns_linear_solvers()
    discretization_controls = (
        discretization_controls or SimpleDiscretizationControls()
    )
    iteration_controls = iteration_controls or SimpleIterationControls(
        max_iterations=1,
        momentum_relative_tolerance=1.0e-3,
        mass_relative_tolerance=1.0e-3,
    )
    pressure_system_controls = (
        pressure_system_controls
        or CollocatedPressureSystemControls(
            pure_neumann_closure=PressureClosureKind.GAUGE
        )
    )
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=case.dirichlet_velocity,
    )
    return CollocatedSimpleSolver(
        diffusion_coef=case.mu,
        convection_coef=case.rho if convection_coef is None else convection_coef,
        source=case.source,
        boundary_conditions=resolve_simple_boundary_conditions(
            mesh,
            boundary,
            discretization_controls,
            pressure_system_controls,
        ),
        discretization_controls=discretization_controls,
        iteration_controls=iteration_controls,
        linear_solvers=linear_solvers,
    )


def test_collocated_simple_solver_default_pressure_relaxation_is_conservative():
    from fealpy.fvm import SimpleIterationControls

    assert SimpleIterationControls().pressure_relaxation == pytest.approx(0.3)


def test_collocated_simple_solver_uses_resolved_geometry_as_single_source():
    solver = _cavity_solver()

    assert not hasattr(solver, "mesh")
    assert not hasattr(solver, "boundary")
    assert not hasattr(solver, "fvm_geometry")
    assert not hasattr(solver, "NC")
    assert not hasattr(solver, "NF")
    assert not hasattr(solver, "GD")
    assert not hasattr(solver, "cm")
    assert (
        solver.spatial_face_velocity.discretization
        is solver.discretization
    )


def test_cell_vector_dof_conversion_is_component_major_on_torch_backend():
    pytest.importorskip("torch")
    from fealpy.backend import backend_manager as bm

    solver = _cavity_solver()
    try:
        bm.set_backend("pytorch")
        cell_vector = bm.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        dofs = solver.discretization.cell_vector_to_dofs(cell_vector)

        assert bm.to_numpy(dofs).tolist() == [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
        assert bm.to_numpy(
            solver.discretization.dofs_to_cell_vector(dofs)
        ).tolist() == [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    finally:
        bm.set_backend("numpy")


def test_collocated_simple_solver_runs_without_model_adapter():
    solver = _cavity_solver()

    result = solver.solve()

    assert result.velocity.shape == (solver.discretization.NC, solver.discretization.GD)
    assert result.pressure.shape == (solver.discretization.NC,)
    assert result.face_velocity.shape == (
        solver.discretization.geometry.mesh.number_of_faces(),
        solver.discretization.GD,
    )
    assert result.face_flux.shape == (solver.discretization.geometry.mesh.number_of_faces(),)
    assert len(result.residual_history) == 1


def test_repeated_simple_solves_are_independent_cold_start_results():
    solver = _cavity_solver(convection_coef=0.0)
    attributes_before = set(vars(solver))

    first = solver.solve()
    first_velocity = first.velocity.copy()
    second = solver.solve()

    assert first is not second
    assert first.residual_history is not second.residual_history
    assert set(vars(solver)) == attributes_before
    np.testing.assert_allclose(first.velocity, second.velocity)
    np.testing.assert_allclose(first.pressure, second.pressure)
    np.testing.assert_allclose(first_velocity, first.velocity)


def test_rhie_chow_requires_physical_cell_velocity_layout():
    from fealpy.backend import backend_manager as bm

    solver = _cavity_solver(convection_coef=0.0)
    algebraic_velocity = bm.zeros(solver.discretization.GD * solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    with pytest.raises(ValueError):
        solver.spatial_face_velocity.interpolate(algebraic_velocity)


def test_simple_reconstructs_stopping_flux_from_updated_state(monkeypatch):
    solver = _cavity_solver(convection_coef=0.0)
    calls = []
    original = solver.rhie_chow_face_velocity

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(solver, "rhie_chow_face_velocity", record)

    result = solver.solve()

    assert len(calls) == 2
    assert result.face_velocity is calls[-1]


def test_simple_always_records_true_momentum_residual_and_termination_state():
    from fealpy.fvm import SimpleIterationControls

    solver = _cavity_solver(
        convection_coef=0.0,
        iteration_controls=SimpleIterationControls(
            max_iterations=1,
            momentum_relative_tolerance=1.0e-30,
            mass_relative_tolerance=1.0e-30,
        ),
    )

    result = solver.solve()

    residual = result.residual_history[-1]
    assert residual.momentum_absolute_l2 >= 0.0
    assert residual.momentum_relative_l2 >= 0.0
    assert result.converged is False
    assert result.termination_reason == "max_iterations"
    assert result.outer_iterations == 1


def test_simple_records_fixed_point_residual_termination():
    from fealpy.fvm import SimpleIterationControls

    solver = _cavity_solver(
        convection_coef=0.0,
        iteration_controls=SimpleIterationControls(
            max_iterations=1,
            momentum_relative_tolerance=1.0e6,
            mass_relative_tolerance=1.0e6,
        ),
    )

    result = solver.solve()

    assert result.converged is True
    assert result.termination_reason == "fixed_point_residuals"
    assert result.outer_iterations == 1


def test_collocated_simple_solver_has_one_pressure_gradient_owner():
    from fealpy.fvm import SimpleDiscretizationControls

    controls = SimpleDiscretizationControls(
        gradient_layer_weights=(1.0, 0.05),
        gradient_boundary_weight=0.75,
    )
    solver = _cavity_solver(discretization_controls=controls)

    assert (
        solver.pressure_gradient
        is solver.rhie_chow.boundary.pressure_state.gradient
    )
    assert solver.pressure_gradient.layer_weights == (1.0, 0.05)
    assert solver.pressure_gradient.boundary_weight == 0.75


def test_steady_momentum_terms_sum_to_production_balance():
    import numpy as np
    from fealpy.backend import backend_manager as bm

    solver = _cavity_solver()
    cell_center = solver.discretization.geometry.cell_center
    velocity = bm.stack(
        (cell_center[:, 0] + cell_center[:, 1], cell_center[:, 0] - cell_center[:, 1]),
        axis=-1,
    )
    pressure = cell_center[:, 0] - 0.25 * cell_center[:, 1]
    face_velocity = solver.spatial_face_velocity.interpolate(velocity)

    from fealpy.fvm.collocated_momentum_equation import (
        MomentumBalance,
        MomentumTerms,
    )

    pressure_gradient = solver.pressure_gradient.cell_gradient(pressure)
    terms = solver.momentum.terms(
        pressure,
        velocity,
        face_velocity,
        pressure_gradient=pressure_gradient,
    )
    assert isinstance(terms, MomentumTerms)
    expected_residual = (
        terms.diffusion
        + terms.convection
        + terms.pressure
        - terms.source
    )
    balance = solver.momentum.balance(
        pressure,
        velocity,
        face_velocity,
        pressure_gradient=pressure_gradient,
    )
    assert isinstance(balance, MomentumBalance)

    np.testing.assert_allclose(
        bm.to_numpy(terms.residual),
        bm.to_numpy(expected_residual),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        bm.to_numpy(balance.residual),
        bm.to_numpy(terms.residual),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_collocated_simple_solver_runs_with_second_order_face_velocity_scheme():
    from fealpy.fvm import SimpleDiscretizationControls, SimpleIterationControls

    solver = _cavity_solver(
        convection_coef=0.0,
        discretization_controls=SimpleDiscretizationControls(
            spatial_face_velocity_scheme="second_order_reconstructed",
        ),
        iteration_controls=SimpleIterationControls(
            max_iterations=1,
            momentum_nonorthogonal_max_iterations=0,
            pressure_nonorthogonal_max_iterations=0,
            momentum_relative_tolerance=1.0e-3,
            mass_relative_tolerance=1.0e-3,
        ),
    )

    result = solver.solve()

    assert solver.spatial_face_velocity.scheme == "second_order_reconstructed"
    assert result.velocity.shape == (solver.discretization.NC, solver.discretization.GD)
    assert result.pressure.shape == (solver.discretization.NC,)


def test_collocated_simple_solver_runs_with_cell_anchored_face_flux_correction(
    monkeypatch,
):
    from fealpy.fvm import SimpleDiscretizationControls, SimpleIterationControls

    solver = _cavity_solver(
        convection_coef=0.0,
        resolution=4,
        discretization_controls=SimpleDiscretizationControls(
            face_flux_correction_scheme="cell_anchored_quadratic",
            face_flux_max_condition=75.0,
        ),
        iteration_controls=SimpleIterationControls(
            max_iterations=1,
            momentum_nonorthogonal_max_iterations=0,
            pressure_nonorthogonal_max_iterations=0,
            momentum_relative_tolerance=1.0e-3,
            mass_relative_tolerance=1.0e-3,
        ),
    )
    captured_boundary_values = []
    operator = solver.spatial_face_velocity
    original = operator.reconstruct

    def record_boundary_values(*args, **kwargs):
        captured_boundary_values.append(
            operator.boundary.dirichlet_face_values
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(operator, "reconstruct", record_boundary_values)

    result = solver.solve()

    face_flux_reconstruct = solver.spatial_face_velocity.face_flux_reconstruct
    assert face_flux_reconstruct.method == "cell_anchored_quadratic"
    assert (
        face_flux_reconstruct.cell_anchored_quadratic.max_condition
        == 75.0
    )
    assert face_flux_reconstruct.diagnostics()["minimum_rank"] == 5
    assert captured_boundary_values
    assert all(
        value is solver.spatial_face_velocity.boundary.dirichlet_face_values
        for value in captured_boundary_values
    )
    assert result.velocity.shape == (solver.discretization.NC, solver.discretization.GD)
    assert result.pressure.shape == (solver.discretization.NC,)


def test_ns_fvm_simple_model_consumes_explicit_profile():
    from dataclasses import replace

    from fealpy.fvm import (
        NSFVMSimpleModel,
        steady_ns_high_accuracy_simple_profile,
    )

    base = steady_ns_high_accuracy_simple_profile()
    profile = replace(
        base,
        discretization=replace(
            base.discretization,
            face_flux_max_stencil_layers=5,
            face_flux_max_condition=75.0,
        ),
    )

    model = NSFVMSimpleModel(
        {
            "pde": 1,
            "mesh_type": "uniform_quad",
            "nx": 4,
            "ny": 4,
            "profile": profile,
            "log_level": "ERROR",
        }
    )

    assert model.profile is profile
    assert (
        model.solver.spatial_face_velocity.face_flux_reconstruct
        .cell_anchored_quadratic.max_condition
        == 75.0
    )


def test_simple_temporary_velocity_reuses_steady_source_rhs(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        CollocatedNSLinearSolvers,
        FVMLinearSolver,
        LinearSolveDiagnostics,
        LinearSolveResult,
        SimpleIterationControls,
    )

    bm.set_backend("numpy")

    linear_solver = FVMLinearSolver("scipy")
    monkeypatch.setattr(
        linear_solver,
        "solve",
        lambda matrix, rhs: LinearSolveResult(
            solution=bm.zeros(
                rhs.shape[0],
                dtype=rhs.dtype,
            ),
            diagnostics=LinearSolveDiagnostics(
                provider="test",
                solver="zero",
                iterations=None,
                converged=True,
                provider_code=None,
                relative_residual=None,
            ),
        ),
    )
    linear_solvers = CollocatedNSLinearSolvers(
        momentum=linear_solver,
        pressure_dirichlet=linear_solver,
        pressure_nullspace=linear_solver,
        pressure_gauge=linear_solver,
    )

    solver = _cavity_solver(
        convection_coef=0.0,
        iteration_controls=SimpleIterationControls(
            max_iterations=1,
            momentum_nonorthogonal_max_iterations=0,
            pressure_nonorthogonal_max_iterations=0,
        ),
        linear_solvers=linear_solvers,
    )
    call_count = 0

    def counted_source_vector(source):
        nonlocal call_count
        call_count += 1
        return bm.ones(solver.discretization.GD * solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)

    monkeypatch.setattr(
        solver.momentum.spatial_operator,
        "source_vector",
        counted_source_vector,
    )

    p = bm.zeros(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    uf = bm.zeros((solver.discretization.geometry.mesh.number_of_faces(), solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    u0 = bm.zeros((solver.discretization.NC, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)

    pressure_gradient = solver.pressure_gradient.cell_gradient(p)
    solver.momentum.predict(
        p,
        uf,
        u0,
        pressure_gradient=pressure_gradient,
        nonorthogonal_tolerance=(
            solver.iteration_controls.momentum_nonorthogonal_rtol
        ),
    )
    solver.momentum.predict(
        p,
        uf,
        u0,
        pressure_gradient=pressure_gradient,
        nonorthogonal_tolerance=(
            solver.iteration_controls.momentum_nonorthogonal_rtol
        ),
    )

    assert call_count == 1


def test_simple_temporary_velocity_returns_named_dual_response():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import SimpleIterationControls
    from fealpy.fvm.collocated_momentum_equation import (
        MomentumPredictorResult,
    )

    bm.set_backend("numpy")
    solver = _cavity_solver(
        convection_coef=0.0,
        iteration_controls=SimpleIterationControls(
            max_iterations=1,
            momentum_nonorthogonal_max_iterations=0,
            pressure_nonorthogonal_max_iterations=0,
        ),
    )
    p = bm.zeros(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    uf = bm.zeros((solver.discretization.geometry.mesh.number_of_faces(), solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    u0 = bm.zeros((solver.discretization.NC, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)

    result = solver.momentum.predict(
        p,
        uf,
        u0,
        pressure_gradient=solver.pressure_gradient.cell_gradient(p),
        nonorthogonal_tolerance=(
            solver.iteration_controls.momentum_nonorthogonal_rtol
        ),
    )

    assert isinstance(result, MomentumPredictorResult)
    assert result.correction_diagonal.shape == (solver.discretization.NC,)
    assert result.spatial_diagonal.shape == (solver.discretization.NC,)
    assert result.cell_velocity.shape == (solver.discretization.NC, solver.discretization.GD)


def test_simple_pressure_gauge_matrix_does_not_use_bilinear_assembly(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import BilinearForm

    solver = _cavity_solver(convection_coef=0.0)
    coef = bm.ones(solver.discretization.geometry.mesh.number_of_faces(), dtype=solver.discretization.geometry.cell_measure.dtype)

    def fail_assembly(self):
        raise AssertionError("pressure gauge matrix should use cached FVM structure")

    monkeypatch.setattr(BilinearForm, "assembly", fail_assembly)

    matrix = solver.pressure_equation.gauge_matrix(coef)

    assert matrix.shape == (solver.discretization.NC + 1, solver.discretization.NC + 1)


def test_simple_pressure_gauge_matrix_matches_bilinear_reference():
    import numpy as np
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import BilinearForm, BlockForm
    from fealpy.sparse import COOTensor
    from fealpy.fvm import ScalarDiffusionIntegrator

    solver = _cavity_solver(convection_coef=0.0)
    coef = bm.linspace(0.3, 1.4, solver.discretization.geometry.mesh.number_of_faces())
    matrix = solver.pressure_equation.gauge_matrix(coef)

    reference = BilinearForm(solver.discretization.space).add_integrator(
        ScalarDiffusionIntegrator(q=2, coef=coef, geometry=solver.discretization.geometry)
    ).assembly()
    gauge_index = bm.stack(
        [
            bm.zeros(solver.discretization.NC, dtype=bm.int32),
            bm.arange(solver.discretization.NC, dtype=bm.int32),
        ],
        axis=0,
    )
    gauge = COOTensor(gauge_index, solver.discretization.geometry.cell_measure, spshape=(1, solver.discretization.NC))
    reference = BlockForm([[reference, gauge.T], [gauge, None]])
    reference = reference.assembly_sparse_matrix(format="csr")

    diff = matrix.to_scipy() - reference.to_scipy()

    np.testing.assert_allclose(diff.data, 0.0, atol=1.0e-13)


def test_scalar_diffusion_matrix_assembler_matches_bilinear_reference():
    import numpy as np
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import BilinearForm
    from fealpy.fvm.scalar_diffusion_integrator import (
        ScalarDiffusionIntegrator,
        ScalarDiffusionMatrixAssembler,
    )

    solver = _cavity_solver(convection_coef=0.0)
    coef = bm.linspace(0.3, 1.4, solver.discretization.geometry.mesh.number_of_faces())
    matrix = ScalarDiffusionMatrixAssembler(
        solver.discretization.space,
        geometry=solver.discretization.geometry,
    ).assembly(coef)

    reference = BilinearForm(solver.discretization.space).add_integrator(
        ScalarDiffusionIntegrator(q=2, coef=coef, geometry=solver.discretization.geometry)
    ).assembly()
    diff = matrix.to_scipy() - reference.to_scipy()

    np.testing.assert_allclose(diff.data, 0.0, atol=1.0e-13)


def test_stokes_simple_model_accepts_explicit_profile():
    from dataclasses import replace

    from fealpy.fvm import (
        StokesFVMSimpleModel,
        steady_ns_high_accuracy_simple_profile,
    )

    base = steady_ns_high_accuracy_simple_profile()
    profile = replace(
        base,
        iteration=replace(
            base.iteration,
            momentum_equation_relaxation=0.6,
        ),
    )

    model = StokesFVMSimpleModel(
        {
            "pde": 1,
            "nx": 2,
            "ny": 2,
            "profile": profile,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )

    assert model.profile is profile
    assert (
        model.solver.iteration_controls.momentum_equation_relaxation
        == 0.6
    )

import numpy as np
import pytest


def _model_options(
    nx=4,
    ny=4,
    time_steps=1,
    *,
    recommended_defaults=False,
):
    from fealpy.fvm import (
        CollocatedPressureSystemControls,
        PressureClosureKind,
    )

    options = {
        "pde": 3,
        "nx": nx,
        "ny": ny,
        "time_steps": time_steps,
        "duration": (0, 1),
        "pbar_log": False,
        "log_level": "WARNING",
    }
    if not recommended_defaults:
        options["pressure_system_controls"] = (
            CollocatedPressureSystemControls(
                pure_neumann_closure=PressureClosureKind.GAUGE,
            )
        )
    return options


def test_piso_model_uses_recommended_default_linear_policy():
    from fealpy.fvm import NSFVMPISOModel, ThirdPartyLinearSolver

    model = NSFVMPISOModel(
        _model_options(nx=2, ny=2, time_steps=1, recommended_defaults=True)
    )

    assert isinstance(
        model.solver.momentum.algebra.linear_solver,
        ThirdPartyLinearSolver,
    )
    assert isinstance(
        model.solver.pressure_system.closure.linear_solver,
        ThirdPartyLinearSolver,
    )
def test_piso_solver_reuses_boundary_geometry():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, time_steps=1))

    assert model.fvm_geometry is model.solver.discretization.geometry


def test_piso_pressure_state_boundary_inherits_diffusion_configuration():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel, PDEBoundaryConditions

    options = _model_options(nx=2, ny=2, time_steps=1)
    options.update(
        {
            "diffusion_method": "bounded_over_relaxed",
            "diffusion_nonorthogonal_eps": 0.1,
            "boundary_conditions": lambda mesh, pde: PDEBoundaryConditions(
                mesh,
                dirichlet_velocity=pde.dirichlet_velocity,
                dirichlet_pressure=lambda p: p[..., 0],
                dirichlet_pressure_selector=lambda p: bm.ones(
                    p.shape[:-1],
                    dtype=bm.bool,
                ),
            ),
        }
    )
    model = NSFVMPISOModel(options)

    pressure_bc = (
        model.solver.pressure_system.boundary.dirichlet_operator
    )
    assert pressure_bc.diffusion_method == "bounded_over_relaxed"
    assert pressure_bc.nonorthogonal_eps == 0.1


def test_collocated_momentum_diffusion_matrix_is_cached():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, time_steps=1))

    spatial = model.solver.momentum.spatial_operator
    assert (
        spatial.diffusion_matrix_template
        is spatial.diffusion_matrix_template
    )


def test_collocated_momentum_time_components_match_cell_diagonal():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, time_steps=1))
    velocity = bm.arange(2 * model.NC, dtype=bm.float64).reshape(model.NC, 2)
    equation = model.solver.momentum
    matrix = equation.time_matrix()
    source = equation.time_source(velocity)
    expected_diagonal = (
        equation.density * model.fvm_geometry.cell_measure / model.solver.controls.tau
    )
    expected_source = (velocity * expected_diagonal[:, None]).flatten(order="F")

    assert np.allclose(
        np.asarray(bm.to_numpy(matrix.diags().values)),
        np.asarray(bm.to_numpy(expected_diagonal)),
    )
    assert np.allclose(np.asarray(bm.to_numpy(source)), np.asarray(bm.to_numpy(expected_source)))


def test_piso_pressure_free_flux_matches_velocity_route():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel
    from fealpy.fvm.collocated_velocity_pressure_coupling import (
        remove_pressure_response,
    )

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, time_steps=1))
    velocity = bm.arange(2 * model.NC, dtype=bm.float64).reshape(model.NC, 2)
    pressure = bm.arange(model.NC, dtype=bm.float64)
    a_p = bm.ones(model.NC)

    pressure_free_velocity, flux = model.solver.pressure_free_flux(velocity, pressure, a_p)
    expected_velocity = remove_pressure_response(
        velocity,
        model.solver.pressure_gradient.cell_gradient(pressure),
        model.solver.discretization.cell_response(a_p),
    )
    expected_flux = model.solver.spatial_face_velocity.compute_flux(
        model.solver.spatial_face_velocity.interpolate(expected_velocity)
    )

    assert np.allclose(np.asarray(bm.to_numpy(pressure_free_velocity)), np.asarray(bm.to_numpy(expected_velocity)))
    assert np.allclose(np.asarray(bm.to_numpy(flux)), np.asarray(bm.to_numpy(expected_flux)))


def test_piso_pressure_correction_step_uses_pressure_free_flux(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel
    import fealpy.fvm.collocated_piso_solver as piso_solver
    model = NSFVMPISOModel(_model_options(nx=2, ny=2, time_steps=1))
    nf = model.mesh.number_of_faces()
    nc = model.NC
    pressure_free_velocity = bm.ones((nc, 2), dtype=model.fvm_geometry.cell_measure.dtype)
    pressure_free_flux = bm.ones(nf, dtype=model.fvm_geometry.cell_measure.dtype) * 0.25
    seen = []

    def record_pressure_free_flux(intermediate_velocity, pressure, a_p):
        seen.append((intermediate_velocity, pressure, a_p))
        return pressure_free_velocity, pressure_free_flux

    monkeypatch.setattr(
        model.solver,
        "pressure_free_flux",
        record_pressure_free_flux,
    )
    monkeypatch.setattr(
        model.solver,
        "transient_face_flux_correction",
        lambda *args, **kwargs: bm.zeros(nf),
    )
    monkeypatch.setattr(
        model.solver.spatial_face_velocity,
        "enforce_boundary_flux",
        lambda flux: flux,
    )
    from fealpy.fvm.collocated_pressure_system import (
        PisoPressureResult,
        PressureFluxParts,
    )

    monkeypatch.setattr(
        model.solver.pressure_system,
        "solve",
        lambda rhs, a_p, **kwargs: PisoPressureResult(
            pressure=bm.zeros(nc),
            pressure_flux=bm.zeros(nf),
            flux_parts=PressureFluxParts(
                orthogonal_flux=bm.zeros(nf),
                cross_flux=bm.zeros(nf),
                boundary_pressure_flux=bm.zeros(nf),
            ),
            face_response_coefficient=bm.zeros(nf),
            nonorthogonal_iterations=0,
            nonorthogonal_residual=None,
            nonorthogonal_relative_update=0.0,
            linear_solves=(),
        ),
    )
    monkeypatch.setattr(
        piso_solver,
        "correct_cell_velocity",
        lambda u_free, pressure_state, a_p, **kwargs: u_free,
    )

    from fealpy.fvm.piso_result import (
        PisoPressureCorrectionStepResult,
    )

    result = model.solver.pressure_correction_step(
        bm.zeros((nc, 2)),
        bm.zeros(nc),
        bm.ones(nc),
        bm.zeros((nc, 2)),
        bm.zeros((nf, 2)),
    )

    assert len(seen) == 1
    assert isinstance(result, PisoPressureCorrectionStepResult)
    assert np.allclose(np.asarray(bm.to_numpy(result.velocity)), np.asarray(bm.to_numpy(pressure_free_velocity)))
    assert np.allclose(np.asarray(bm.to_numpy(result.face_flux)), np.asarray(bm.to_numpy(pressure_free_flux)))
    assert result.diagnostics is None


def test_rhie_chow_face_velocity_uses_resolved_boundary_velocity():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options())
    u = np.ones((model.NC, 2))
    ap = np.ones(model.NC)
    pressure = np.zeros(model.NC)
    response = (
        model.solver.pressure_equation.face_response_coefficient(ap)
    )
    boundary_faces = model.mesh.boundary_face_index()
    face_velocity = model.solver.rhie_chow_face_velocity(
        u,
        pressure,
        response,
    )

    assert np.allclose(
        np.asarray(face_velocity[boundary_faces]),
        np.asarray(
            model.solver.spatial_face_velocity.boundary
            .dirichlet_face_values
        ),
    )


def test_piso_transient_flux_correction_uses_limited_ddtcorr(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options())
    nf = model.mesh.number_of_faces()
    old_face_flux = bm.array(np.linspace(1.0, 2.0, nf))
    old_cell_flux = old_face_flux - 0.25
    response = bm.ones(nf) * 2.0
    face_flux_returns = [old_face_flux, old_cell_flux]

    monkeypatch.setattr(
        model.solver.spatial_face_velocity,
        "compute_flux",
        lambda face_velocity: face_flux_returns.pop(0),
    )
    monkeypatch.setattr(
        model.solver.spatial_face_velocity,
        "interpolate",
        lambda cell_velocity: bm.zeros((nf, 2), dtype=model.fvm_geometry.cell_measure.dtype),
    )
    monkeypatch.setattr(
        model.solver.pressure_equation,
        "face_response_coefficient",
        lambda a_p: response,
    )

    correction = model.solver.transient_face_flux_correction(
        bm.zeros((model.NC, 2)),
        bm.zeros((nf, 2)),
        bm.ones(model.NC),
    )

    boundary = np.asarray(
        bm.to_numpy(
            model.solver.discretization.geometry.face_to_cell[:, 0]
            == model.solver.discretization.geometry.face_to_cell[:, 1]
        )
    )
    flux_correction = np.asarray(bm.to_numpy(old_face_flux - old_cell_flux))
    coeff = 1.0 - np.minimum(
        np.abs(flux_correction) / np.asarray(bm.to_numpy(np.abs(old_face_flux))),
        1.0,
    )
    coeff[boundary] = 0.0
    expected = (
        np.asarray(bm.to_numpy(response))
        * coeff
        * flux_correction
        / model.solver.controls.tau
    )

    assert face_flux_returns == []
    assert np.allclose(np.asarray(bm.to_numpy(correction)), expected)
    assert np.allclose(np.asarray(bm.to_numpy(correction))[boundary], 0.0)


def test_piso_rejects_unknown_options():
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, time_steps=1)
    options["momentum_explicit_correction"] = "openfoam"

    with pytest.raises(
        ValueError,
        match="unsupported NSFVMPISOModel options: momentum_explicit_correction",
    ):
        NSFVMPISOModel(options)


def test_piso_snapshot_callback_respects_interval_and_start_step():
    from fealpy.fvm import NSFVMPISOModel, PisoSnapshot

    options = _model_options(nx=2, ny=2, time_steps=4)
    options.update(
        {
            "snapshot_interval": 2,
            "snapshot_start_step": 2,
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    snapshots = []

    model.solve(snapshot_callback=snapshots.append)

    assert [snapshot.step for snapshot in snapshots] == [2, 4]
    assert all(isinstance(snapshot, PisoSnapshot) for snapshot in snapshots)
    assert all(not hasattr(snapshot, "solver") for snapshot in snapshots)
    assert snapshots[-1].velocity.shape == (model.NC, model.GD)
    assert snapshots[-1].face_velocity.shape[1] == model.GD
    assert snapshots[-1].pressure.shape == (model.NC,)
    assert snapshots[-1].face_flux.shape == (
        snapshots[-1].face_velocity.shape[0],
    )


def test_piso_face_interpolation_option_reaches_momentum_convection(monkeypatch):
    from fealpy.fvm import NSFVMPISOModel
    import fealpy.fvm.collocated_momentum_equation as momentum_equation

    seen = []
    original = momentum_equation.ConvectionMatrixAssembler

    class RecordingConvectionMatrixAssembler(original):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("interpolation"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        momentum_equation,
        "ConvectionMatrixAssembler",
        RecordingConvectionMatrixAssembler,
    )
    options = _model_options(nx=2, ny=2, time_steps=1)
    options.update(
        {
            "momentum_face_interpolation": "linear",
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    U0, Uf0, p0 = model.initial_solution()

    model.solver.momentum.predict(
        U0,
        Uf0,
        p0,
        time=model.solver.controls.tau,
        pressure_gradient=model.solver.pressure_gradient.cell_gradient(p0),
    )

    assert seen == ["linear"]
    assert (
        model.solver.momentum.spatial_operator.face_interpolation
        == "linear"
    )


def test_piso_face_interpolation_option_reaches_pressure_response(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel
    import fealpy.fvm.collocated_pressure_equation as pressure_equation_module

    options = _model_options(nx=2, ny=2, time_steps=1)
    options["pressure_response_interpolation"] = "linear"
    model = NSFVMPISOModel(options)
    seen = []
    nf = model.mesh.number_of_faces()

    def record_scalar(cell_values, method=None):
        seen.append(method)
        return bm.zeros(nf, dtype=cell_values.dtype)

    monkeypatch.setattr(
        pressure_equation_module,
        "interpolate_cell_to_face",
        lambda values, *, geometry, method: record_scalar(
            values,
            method=method,
        ),
    )

    model.solver.pressure_equation.face_response_coefficient(
        bm.ones(model.NC),
    )

    assert seen == ["linear"]


def test_piso_face_interpolation_option_reaches_pressure_free_flux(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel
    import fealpy.fvm.collocated_piso_solver as piso_solver
    options = _model_options(nx=2, ny=2, time_steps=1)
    options["rhie_chow_velocity_interpolation"] = "linear"
    model = NSFVMPISOModel(options)
    seen = []
    nf = model.mesh.number_of_faces()

    def record_vector(cell_velocity, method=None):
        seen.append(method)
        return bm.zeros((nf, 2), dtype=model.fvm_geometry.cell_measure.dtype)

    monkeypatch.setattr(
        model.solver.spatial_face_velocity,
        "interpolate",
        lambda cell_velocity: record_vector(
            cell_velocity,
            method=model.solver.spatial_face_velocity.interpolation,
        ),
    )
    monkeypatch.setattr(
        piso_solver,
        "remove_pressure_response",
        lambda intermediate_velocity, pressure, a_p, **kwargs: (
            intermediate_velocity
        ),
    )
    monkeypatch.setattr(
        model.solver.spatial_face_velocity,
        "compute_flux",
        lambda face_velocity: bm.zeros(nf),
    )
    monkeypatch.setattr(
        model.solver,
        "transient_face_flux_correction",
        lambda *args, **kwargs: bm.zeros(nf),
    )
    monkeypatch.setattr(
        model.solver.spatial_face_velocity,
        "enforce_boundary_flux",
        lambda flux: flux,
    )
    from fealpy.fvm.collocated_pressure_system import (
        PisoPressureResult,
        PressureFluxParts,
    )

    monkeypatch.setattr(
        model.solver.pressure_system,
        "solve",
        lambda rhs, a_p, **kwargs: PisoPressureResult(
            pressure=bm.zeros(model.NC),
            pressure_flux=bm.zeros(nf),
            flux_parts=PressureFluxParts(
                orthogonal_flux=bm.zeros(nf),
                cross_flux=bm.zeros(nf),
                boundary_pressure_flux=bm.zeros(nf),
            ),
            face_response_coefficient=bm.zeros(nf),
            nonorthogonal_iterations=0,
            nonorthogonal_residual=None,
            nonorthogonal_relative_update=0.0,
            linear_solves=(),
        ),
    )
    monkeypatch.setattr(
        piso_solver,
        "correct_cell_velocity",
        lambda pressure_free_velocity, pressure_state, a_p, **kwargs: (
            pressure_free_velocity
        ),
    )

    model.solver.pressure_correction_step(
        bm.zeros((model.NC, 2)),
        bm.zeros(model.NC),
        bm.ones(model.NC),
        bm.zeros((model.NC, 2)),
        bm.zeros((nf, 2)),
    )

    assert seen == ["linear"]


def test_piso_momentum_nonorthogonal_correction_uses_picard_loop(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, time_steps=1)
    options.update(
        {
            "momentum_nonorthogonal_max_iterations": 2,
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    U0, Uf0, p0 = model.initial_solution()
    solves = []
    linear_solver = model.solver.momentum.algebra.linear_solver
    original_solve = linear_solver.solve

    def solve_momentum(matrix, rhs):
        solves.append(1)
        return original_solve(matrix, rhs)

    monkeypatch.setattr(linear_solver, "solve", solve_momentum)

    predictor = model.solver.momentum.predict(
        U0,
        Uf0,
        p0,
        time=model.solver.controls.tau,
        pressure_gradient=model.solver.pressure_gradient.cell_gradient(p0),
    )
    U = predictor.cell_velocity

    assert U.shape == U0.shape
    assert len(solves) == model.GD * (
        1 + predictor.nonorthogonal_iterations
    )
    assert predictor.nonorthogonal_iterations <= 2
    assert (
        predictor.nonorthogonal_residual.relative
        <= model.solver.controls.momentum_nonorthogonal_rtol
    )


def test_piso_zero_momentum_nonorthogonal_still_solves_base_equation(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, time_steps=1)
    options.update(
        {
            "momentum_nonorthogonal_max_iterations": 0,
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    U0, Uf0, p0 = model.initial_solution()
    solves = []

    def forbidden_cross_source(velocity):
        raise AssertionError("zero nonorthogonal corrections should not build cross RHS")

    def solve_momentum(matrix, rhs):
        from fealpy.fvm import LinearSolveDiagnostics, LinearSolveResult

        solves.append(1)
        return LinearSolveResult(
            solution=bm.ones(model.NC, dtype=U0.dtype),
            diagnostics=LinearSolveDiagnostics(
                provider="test",
                solver="ones",
                iterations=None,
                converged=True,
                provider_code=None,
                relative_residual=0.0,
            ),
        )

    monkeypatch.setattr(
        model.solver.momentum.spatial_operator,
        "nonorthogonal_rhs",
        forbidden_cross_source,
    )
    monkeypatch.setattr(
        model.solver.momentum.algebra.linear_solver,
        "solve",
        solve_momentum,
    )

    predictor = model.solver.momentum.predict(
        U0,
        Uf0,
        p0,
        time=model.solver.controls.tau,
        pressure_gradient=model.solver.pressure_gradient.cell_gradient(p0),
    )
    U = predictor.cell_velocity

    assert len(solves) == model.GD
    assert predictor.nonorthogonal_iterations == 0
    assert np.allclose(np.asarray(bm.to_numpy(U)), 1.0)


def test_piso_pressure_nonorthogonal_stops_on_complete_residual():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, time_steps=1)
    options["pressure_nonorthogonal_max_iterations"] = 2
    model = NSFVMPISOModel(options)
    nc = model.NC

    result = model.solver.pressure_system.solve(
        bm.zeros(nc, dtype=model.fvm_geometry.cell_measure.dtype),
        bm.ones(nc, dtype=model.fvm_geometry.cell_measure.dtype),
        initial_pressure_state=bm.zeros(nc, dtype=model.fvm_geometry.cell_measure.dtype),
    )

    assert np.allclose(np.asarray(bm.to_numpy(result.pressure)), 0.0)
    assert result.nonorthogonal_iterations == 0
    assert result.nonorthogonal_residual.relative == 0.0
    assert result.flux_parts.cross_flux.shape == (
        model.mesh.number_of_faces(),
    )


def test_piso_pressure_nonorthogonal_first_rhs_uses_entering_pressure(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel
    from fealpy.fvm.solver_diagnostics import EquationResidual
    import fealpy.fvm.collocated_pressure_system as pressure_system_module

    options = _model_options(nx=2, ny=2, time_steps=1)
    options["pressure_nonorthogonal_max_iterations"] = 2
    model = NSFVMPISOModel(options)
    nf = model.mesh.number_of_faces()
    nc = model.NC
    captured_cross_rhs = []
    solve_values = []

    class FakeSolver:
        def solve(self, matrix, rhs):
            from fealpy.fvm import (
                LinearSolveDiagnostics,
                LinearSolveResult,
            )

            pressure_value = float(len(solve_values) + 2)
            solve_values.append(pressure_value)
            pressure = (
                bm.ones(nc, dtype=model.fvm_geometry.cell_measure.dtype)
                * pressure_value
            )
            return LinearSolveResult(
                solution=bm.concatenate(
                    [
                        pressure,
                        bm.zeros(
                            1,
                            dtype=model.fvm_geometry.cell_measure.dtype,
                        ),
                    ]
                ),
                diagnostics=LinearSolveDiagnostics(
                    provider="test",
                    solver="sequence",
                    iterations=None,
                    converged=True,
                    provider_code=None,
                    relative_residual=0.0,
                ),
            )

    model.solver.pressure_system.closure.linear_solver = FakeSolver()
    equation = model.solver.pressure_equation
    monkeypatch.setattr(
        equation,
        "divergence_from_flux",
        lambda flux: bm.ones(nc, dtype=model.fvm_geometry.cell_measure.dtype) * flux[0],
    )
    monkeypatch.setattr(
        equation,
        "nonorthogonal_cross_flux",
        lambda pressure, coef, *, interpolation_method,
        gradient_boundary, pressure_gradient: (
            bm.ones(nf, dtype=model.fvm_geometry.cell_measure.dtype) * pressure[0]
        ),
    )
    monkeypatch.setattr(
        equation,
        "orthogonal_flux",
        lambda pressure, coef: bm.zeros(nf, dtype=model.fvm_geometry.cell_measure.dtype),
    )
    monkeypatch.setattr(
        equation,
        "add_dirichlet_flux",
        lambda flux, pressure, coef, faces, values: flux,
    )
    closure = model.solver.pressure_system.closure
    original_rhs = closure.rhs

    def record_rhs(rhs, cross_rhs, coef):
        captured_cross_rhs.append(np.asarray(bm.to_numpy(cross_rhs)).copy())
        return original_rhs(rhs, cross_rhs, coef)

    monkeypatch.setattr(closure, "rhs", record_rhs)
    monkeypatch.setattr(
        pressure_system_module,
        "normalized_equation_residual",
        lambda lhs, rhs: EquationResidual(
            absolute=0.0,
            relative=0.0,
            scale=1.0,
        ),
    )

    entering_pressure = bm.ones(nc, dtype=model.fvm_geometry.cell_measure.dtype) * 7.0
    model.solver.pressure_system.solve(
        bm.zeros(nc, dtype=model.fvm_geometry.cell_measure.dtype),
        bm.ones(nc, dtype=model.fvm_geometry.cell_measure.dtype),
        initial_pressure_state=entering_pressure,
    )

    assert solve_values == [2.0]
    assert np.allclose(captured_cross_rhs[0], 7.0)
    assert np.allclose(captured_cross_rhs[1], 2.0)


def test_piso_zero_pressure_nonorthogonal_solves_base_system_once(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, time_steps=1)
    options["pressure_nonorthogonal_max_iterations"] = 0
    model = NSFVMPISOModel(options)
    nc = model.NC
    nf = model.mesh.number_of_faces()
    assembled_cross_rhs = []
    solves = []

    class FakeSolver:
        def solve(self, matrix, rhs):
            from fealpy.fvm import (
                LinearSolveDiagnostics,
                LinearSolveResult,
            )

            solves.append(1)
            return LinearSolveResult(
                solution=bm.concatenate(
                    [
                        bm.ones(
                            nc,
                            dtype=model.fvm_geometry.cell_measure.dtype,
                        )
                        * 3.0,
                        bm.zeros(
                            1,
                            dtype=model.fvm_geometry.cell_measure.dtype,
                        ),
                    ]
                ),
                diagnostics=LinearSolveDiagnostics(
                    provider="test",
                    solver="constant",
                    iterations=None,
                    converged=True,
                    provider_code=None,
                    relative_residual=0.0,
                ),
            )

    model.solver.pressure_system.closure.linear_solver = FakeSolver()
    equation = model.solver.pressure_equation
    monkeypatch.setattr(
        equation,
        "nonorthogonal_cross_flux",
        lambda pressure, coef, *, interpolation_method,
        gradient_boundary, pressure_gradient: (_ for _ in ()).throw(
            AssertionError("zero nonorthogonal corrections should not build cross flux")
        ),
    )
    monkeypatch.setattr(
        equation,
        "orthogonal_flux",
        lambda pressure, coef: bm.ones(nf, dtype=model.fvm_geometry.cell_measure.dtype),
    )
    monkeypatch.setattr(
        equation,
        "add_dirichlet_flux",
        lambda flux, pressure, coef, faces, values: flux,
    )

    closure = model.solver.pressure_system.closure
    original_rhs = closure.rhs

    def record_rhs(rhs, cross_rhs, coef):
        assembled_cross_rhs.append(np.asarray(bm.to_numpy(cross_rhs)).copy())
        return original_rhs(rhs, cross_rhs, coef)

    monkeypatch.setattr(closure, "rhs", record_rhs)

    result = model.solver.pressure_system.solve(
        bm.zeros(nc, dtype=model.fvm_geometry.cell_measure.dtype),
        bm.ones(nc, dtype=model.fvm_geometry.cell_measure.dtype),
        initial_pressure_state=bm.ones(nc, dtype=model.fvm_geometry.cell_measure.dtype),
    )

    assert len(solves) == 1
    assert result.nonorthogonal_iterations == 0
    assert np.allclose(assembled_cross_rhs[0], 0.0)
    assert np.allclose(np.asarray(bm.to_numpy(result.pressure)), 3.0)
    assert np.allclose(
        np.asarray(bm.to_numpy(result.pressure_flux)),
        1.0,
    )


def test_piso_corrector_diagnostics_can_be_enabled_without_callback():
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, time_steps=1)
    options.update(
        {
            "diagnostics_enabled": True,
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)

    result = model.solve()

    assert (
        len(result.corrector_diagnostics)
        == model.solver.controls.n_correctors
    )
    first = result.corrector_diagnostics[0]
    assert first.step == 1
    assert first.corrector == 1
    assert (
        first.pressure_correction.pressure_free_divergence_linf
        >= 0.0
    )
    assert first.rhie_chow_flux_error_linf >= 0.0


def test_piso_corrector_callback_still_enables_diagnostics():
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, time_steps=1)
    options.update({"log_level": "ERROR"})
    model = NSFVMPISOModel(options)
    rows = []

    result = model.solve(corrector_callback=rows.append)

    assert len(rows) == model.solver.controls.n_correctors
    assert list(result.corrector_diagnostics) == rows
    assert rows[0].step == 1
    assert rows[0].corrector == 1


def test_piso_pressure_step_skips_diagnostic_only_divergence(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(
        _model_options(nx=2, ny=2, time_steps=1)
    )
    solver = model.solver
    nc = solver.discretization.NC
    nf = solver.discretization.NF
    calls = 0
    original = solver.pressure_equation.divergence_from_flux

    def count(face_flux):
        nonlocal calls
        calls += 1
        return original(face_flux)

    monkeypatch.setattr(
        solver.pressure_equation,
        "divergence_from_flux",
        count,
    )
    solver.pressure_correction_step(
        bm.zeros((nc, solver.discretization.GD)),
        bm.zeros(nc),
        bm.ones(nc),
        bm.zeros((nc, solver.discretization.GD)),
        bm.zeros((nf, solver.discretization.GD)),
    )
    calls_without_diagnostics = calls
    calls = 0
    solver.pressure_correction_step(
        bm.zeros((nc, solver.discretization.GD)),
        bm.zeros(nc),
        bm.ones(nc),
        bm.zeros((nc, solver.discretization.GD)),
        bm.zeros((nf, solver.discretization.GD)),
        return_diagnostics=True,
    )

    assert calls == calls_without_diagnostics + 1

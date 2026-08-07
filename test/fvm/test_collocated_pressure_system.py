def _solver():
    from test.fvm.test_collocated_simple_solver_structure import _cavity_solver

    return _cavity_solver(convection_coef=0.0)


def test_nullspace_closure_returns_decoded_call_local_result():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm.collocated_pressure_system import (
        NullspacePressureSystemClosure,
    )
    from fealpy.fvm.fvm_linear_solver import (
        LinearSolveDiagnostics,
        LinearSolveResult,
    )

    calls = []

    class Equation:
        def zero_mean_pressure(self, pressure):
            calls.append(tuple(bm.to_numpy(pressure)))
            return pressure - bm.mean(pressure)

    class LinearSolver:
        def solve(self, matrix, rhs):
            return LinearSolveResult(
                solution=bm.array([3.0, 5.0]),
                diagnostics=LinearSolveDiagnostics(
                    provider="petsc",
                    solver="gmres-hypre",
                    iterations=4,
                    converged=True,
                    provider_code=2,
                    relative_residual=1.0e-12,
                ),
            )

    closure = NullspacePressureSystemClosure(
        Equation(),
        LinearSolver(),
    )
    result = closure.solve("matrix", bm.array([1.0, -1.0]))

    assert isinstance(result, LinearSolveResult)
    assert bm.to_numpy(result.solution).tolist() == [-1.0, 1.0]
    assert result.diagnostics.iterations == 4
    assert calls == [(3.0, 5.0)]


def test_pressure_nonorthogonal_result_owns_linear_diagnostics():
    from dataclasses import fields

    from fealpy.fvm.collocated_pressure_system import (
        NonorthogonalPressureResult,
        PressureCorrectionResult,
        PisoPressureResult,
    )

    for result_type in (
        NonorthogonalPressureResult,
        PressureCorrectionResult,
        PisoPressureResult,
    ):
        assert "linear_solves" in {
            field.name for field in fields(result_type)
        }


def test_simple_pressure_solve_returns_call_local_nonorthogonal_state():
    from dataclasses import fields

    from fealpy.backend import backend_manager as bm
    from fealpy.fvm.collocated_pressure_system import (
        PressureCorrectionResult,
        SimplePressureCorrectionSystem,
    )

    solver = _solver()
    system = solver.pressure_system
    assert isinstance(system, SimplePressureCorrectionSystem)
    face_velocity = bm.zeros(
        (solver.discretization.NF, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    result = system.solve(
        face_velocity,
        bm.ones(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype),
    )

    assert isinstance(result, PressureCorrectionResult)
    assert result.pressure_correction.shape == (solver.discretization.NC,)
    assert "face_response_coefficient" not in {
        field.name for field in fields(PressureCorrectionResult)
    }
    assert result.nonorthogonal_iterations >= 0


def test_pressure_system_closure_derives_tensor_metadata_from_equation():
    import inspect
    from fealpy.fvm.collocated_pressure_system import (
        GaugePressureSystemClosure,
        resolve_pressure_system_closure,
    )

    assert "dtype" not in inspect.signature(
        GaugePressureSystemClosure
    ).parameters
    assert "dtype" not in inspect.signature(
        resolve_pressure_system_closure
    ).parameters


def test_gauge_pressure_encoding_inherits_reference_tensor_context():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm.collocated_pressure_system import (
        GaugePressureSystemClosure,
    )

    pressure = bm.array([1.0, 2.0])
    closure = object.__new__(GaugePressureSystemClosure)

    encoded = closure.encode(pressure)

    assert encoded.dtype == pressure.dtype
    assert bm.get_device(encoded) == bm.get_device(pressure)

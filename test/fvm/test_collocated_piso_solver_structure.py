import numpy as np


def _piso_model():
    from fealpy.fvm import (
        CollocatedPressureSystemControls,
        NSFVMPISOModel,
        PressureClosureKind,
    )

    return NSFVMPISOModel(
        {
            "pde": 3,
            "nx": 2,
            "ny": 2,
            "time_steps": 1,
            "duration": (0.0, 1.0),
            "diagnostics_enabled": True,
            "pressure_system_controls": (
                CollocatedPressureSystemControls(
                    pure_neumann_closure=PressureClosureKind.GAUGE,
                )
            ),
            "pbar_log": False,
            "log_level": "ERROR",
        }
    )


def test_piso_model_composes_solver_instead_of_inheriting_it():
    from fealpy.fvm import CollocatedPisoSolver, NSFVMPISOModel

    model = _piso_model()

    assert not issubclass(NSFVMPISOModel, CollocatedPisoSolver)
    assert isinstance(model.solver, CollocatedPisoSolver)


def test_piso_solver_uses_resolved_geometry_as_single_source():
    solver = _piso_model().solver

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


def test_repeated_piso_solves_return_independent_results_without_solver_state():
    from fealpy.fvm import PisoSolveResult

    model = _piso_model()
    attributes_before = set(vars(model.solver))

    first = model.solve()
    first_velocity = first.velocity.copy()
    second = model.solve()

    assert isinstance(first, PisoSolveResult)
    assert first is not second
    assert first.corrector_diagnostics is not second.corrector_diagnostics
    assert set(vars(model.solver)) == attributes_before
    assert not hasattr(model.solver, "velocity")
    assert not hasattr(model.solver, "pressure")
    assert not hasattr(model.solver, "face_velocity")
    assert not hasattr(model.solver, "face_flux")
    assert not hasattr(model.solver, "corrector_diagnostics")
    np.testing.assert_allclose(first.velocity, second.velocity)
    np.testing.assert_allclose(first.pressure, second.pressure)
    np.testing.assert_allclose(first_velocity, first.velocity)


def test_piso_model_error_consumes_explicit_result():
    model = _piso_model()
    result = model.solve()

    errors = model.compute_error(result)

    assert len(errors) == model.GD + 1

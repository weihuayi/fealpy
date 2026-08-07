import numpy as np

from fealpy.backend import backend_manager as bm


def _simple_solver(*, face_flux_correction_scheme="none"):
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
    mesh = case.init_mesh["uniform_quad"](nx=3, ny=3)
    discretization_controls = SimpleDiscretizationControls(
        face_flux_correction_scheme=face_flux_correction_scheme,
    )
    boundary = resolve_simple_boundary_conditions(
        mesh,
        PDEBoundaryConditions(
            mesh,
            dirichlet_velocity=case.dirichlet_velocity,
        ),
        discretization_controls,
        CollocatedPressureSystemControls(
            pure_neumann_closure=PressureClosureKind.GAUGE
        ),
    )
    return CollocatedSimpleSolver(
        diffusion_coef=case.mu,
        convection_coef=0.0,
        source=case.source,
        boundary_conditions=boundary,
        discretization_controls=discretization_controls,
        iteration_controls=SimpleIterationControls(
            max_iterations=1,
            momentum_relative_tolerance=1.0e-3,
            mass_relative_tolerance=1.0e-3,
        ),
        linear_solvers=build_collocated_ns_linear_solvers(),
    )


def test_second_order_spatial_reconstruction_recovers_affine_velocity(
    monkeypatch,
):
    solver = _simple_solver()
    cell_center = solver.discretization.geometry.cell_center
    face_center = solver.discretization.geometry.face_center
    velocity = bm.stack(
        (
            cell_center[:, 0] + 2.0 * cell_center[:, 1],
            -cell_center[:, 0] + 3.0 * cell_center[:, 1],
        ),
        axis=-1,
    )
    exact_gradient = bm.broadcast_to(
        bm.array([[1.0, 2.0], [-1.0, 3.0]])[None, :, :],
        (solver.discretization.NC, solver.discretization.GD, solver.discretization.GD),
    )
    monkeypatch.setattr(
        solver.spatial_face_velocity.boundary.gradient.lsq_reconstruct,
        "layered_lsq",
        lambda value: exact_gradient,
    )

    actual = solver.spatial_face_velocity.second_order_reconstruct(
        velocity
    )
    expected = bm.stack(
        (
            face_center[:, 0] + 2.0 * face_center[:, 1],
            -face_center[:, 0] + 3.0 * face_center[:, 1],
        ),
        axis=-1,
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_configured_reconstruction_delegates_to_scheme_operator(monkeypatch):
    solver = _simple_solver()
    operator = solver.spatial_face_velocity
    velocity = bm.zeros(
        (solver.discretization.NC, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    calls = []
    original = operator.second_order_reconstruct

    def record(cell_velocity):
        calls.append(cell_velocity)
        return original(cell_velocity)

    monkeypatch.setattr(operator, "second_order_reconstruct", record)

    operator.reconstruct(velocity)

    assert calls == [velocity]


def test_rhie_chow_applies_only_pressure_gradient_difference(monkeypatch):
    solver = _simple_solver()
    base = bm.ones((solver.discretization.NF, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    pressure = bm.zeros(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    response = bm.ones(solver.discretization.NF, dtype=solver.discretization.geometry.cell_measure.dtype)
    pressure_gradient = bm.zeros(
        (solver.discretization.NC, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    difference = bm.ones_like(base) * 0.25
    monkeypatch.setattr(
        solver.rhie_chow,
        "pressure_gradient_difference",
        lambda p, pressure_gradient: difference,
    )

    actual = solver.rhie_chow.apply(
        base,
        pressure,
        response,
        pressure_gradient,
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(base - 0.25),
    )


def test_coupling_uses_canonical_scalar_cell_response():
    from fealpy.fvm.collocated_velocity_pressure_coupling import (
        correct_cell_velocity,
    )

    solver = _simple_solver()
    velocity = bm.ones((solver.discretization.NC, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    diagonal = 2.0 * solver.discretization.geometry.cell_measure
    gradient = bm.ones_like(velocity)
    cell_response = solver.discretization.cell_response(diagonal)

    actual = correct_cell_velocity(
        velocity,
        gradient,
        cell_response,
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(velocity - 0.5 * gradient),
    )


def test_face_flux_correction_computes_current_flux_once(monkeypatch):
    solver = _simple_solver(
        face_flux_correction_scheme="cell_anchored_quadratic",
    )
    operator = solver.spatial_face_velocity
    velocity = bm.zeros(
        (solver.discretization.NC, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    calls = 0
    original = operator.compute_flux

    def count(face_velocity):
        nonlocal calls
        calls += 1
        return original(face_velocity)

    monkeypatch.setattr(operator, "compute_flux", count)

    operator.reconstruct(velocity)

    assert calls == 1


def test_rhie_chow_owns_only_canonical_geometry_binding():
    import inspect
    import typing

    from fealpy.fvm.collocated_face_velocity_reconstruct import (
        RhieChowInterpolation,
    )

    solver = _simple_solver()
    rhie_chow = solver.rhie_chow
    hints = typing.get_type_hints(RhieChowInterpolation.__init__)

    assert rhie_chow.geometry is solver.discretization.geometry
    assert not hasattr(rhie_chow, "mesh")
    assert not hasattr(rhie_chow, "fvm_geometry")
    assert not hasattr(rhie_chow, "cm")
    assert not hasattr(rhie_chow, "NC")
    assert not hasattr(rhie_chow, "GD")
    assert not hasattr(rhie_chow, "face_to_cell")
    assert not hasattr(rhie_chow, "d_f")
    assert not hasattr(rhie_chow, "mag_d_f")
    assert not hasattr(
        RhieChowInterpolation,
        "apply_dirichlet_pressure_boundary_partial",
    )
    assert set(inspect.signature(RhieChowInterpolation).parameters) <= set(
        hints
    )
    assert hints["return"] is type(None)

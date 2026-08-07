import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import PressureClosureKind


def _zero_velocity(points):
    return bm.zeros(points.shape, dtype=points.dtype)


def _right_outlet(points):
    return bm.abs(points[..., 0] - 1.0) < 1.0e-12


def _fixed_velocity_boundary(points):
    return (
        (bm.abs(points[..., 0]) < 1.0e-12)
        | (bm.abs(points[..., 1]) < 1.0e-12)
        | (bm.abs(points[..., 1] - 1.0) < 1.0e-12)
    )


def _resolved_traction_faces(solver):
    traction = solver.momentum.spatial_operator.momentum_boundary.traction
    if traction is None:
        return bm.nonzero(solver.discretization.geometry.is_boundary)[0][:0]
    return traction.faces


def _operator_solver(
    *,
    traction=None,
    convection_coef=0.0,
):
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        CollocatedPressureSystemControls,
        CollocatedSimpleSolver,
        EngineeringBoundaryConditions,
        PressureClosureKind,
        SimpleDiscretizationControls,
        SimpleIterationControls,
        build_collocated_ns_linear_solvers,
        resolve_simple_boundary_conditions,
    )
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    conditions = [
        BoundaryCondition(
            "velocity",
            "fixed",
            "dirichlet",
            _zero_velocity,
        ),
        BoundaryCondition("velocity", "outlet", "natural", None),
    ]
    if traction is not None:
        conditions.append(
            BoundaryCondition("momentum", "outlet", "traction", traction)
        )
    engineering = EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch("fixed", _fixed_velocity_boundary),
            BoundaryPatch("outlet", _right_outlet),
        ],
        conditions=conditions,
    )
    discretization = SimpleDiscretizationControls()
    iteration = SimpleIterationControls(
        momentum_nonorthogonal_max_iterations=0,
        pressure_nonorthogonal_max_iterations=0,
    )
    pressure_system = CollocatedPressureSystemControls(
        pure_neumann_closure=PressureClosureKind.GAUGE,
    )
    return CollocatedSimpleSolver(
        diffusion_coef=1.0,
        convection_coef=convection_coef,
        source=lambda points: bm.zeros(points.shape, dtype=points.dtype),
        boundary_conditions=resolve_simple_boundary_conditions(
            mesh,
            engineering,
            discretization,
            pressure_system,
        ),
        discretization_controls=discretization,
        iteration_controls=iteration,
        linear_solvers=build_collocated_ns_linear_solvers(),
    )


def _linear_probe_solver(*, traction):
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        CollocatedPressureSystemControls,
        CollocatedSimpleSolver,
        EngineeringBoundaryConditions,
        SimpleDiscretizationControls,
        SimpleIterationControls,
        build_collocated_ns_linear_solvers,
        resolve_simple_boundary_conditions,
    )
    from fealpy.mesh import QuadrangleMesh

    viscosity = 1.0
    rate = 0.1
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=4, ny=4)

    def velocity(points):
        return bm.stack(
            [
                rate * points[..., 0],
                -rate * points[..., 1],
            ],
            axis=-1,
        )

    def source(points):
        return bm.stack(
            [
                rate**2 * points[..., 0],
                rate**2 * points[..., 1],
            ],
            axis=-1,
        )

    conditions = [
        BoundaryCondition("velocity", "fixed", "dirichlet", velocity),
        BoundaryCondition("velocity", "outlet", "natural", None),
    ]
    if traction:
        conditions.append(
            BoundaryCondition(
                "momentum",
                "outlet",
                "traction",
                (0.0, 0.0),
            )
        )
    else:
        conditions.append(
            BoundaryCondition(
                "pressure",
                "outlet",
                "dirichlet",
                viscosity * rate,
            )
        )
    engineering = EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch("fixed", _fixed_velocity_boundary),
            BoundaryPatch("outlet", _right_outlet),
        ],
        conditions=conditions,
    )
    discretization = SimpleDiscretizationControls()
    iteration = SimpleIterationControls(
        momentum_equation_relaxation=1.0,
        momentum_nonorthogonal_max_iterations=1,
        momentum_nonorthogonal_rtol=1.0e-12,
        pressure_nonorthogonal_max_iterations=0,
    )
    pressure_system = CollocatedPressureSystemControls(
        pure_neumann_closure=PressureClosureKind.GAUGE,
    )
    solver = CollocatedSimpleSolver(
        diffusion_coef=viscosity,
        convection_coef=1.0,
        source=source,
        boundary_conditions=resolve_simple_boundary_conditions(
            mesh,
            engineering,
            discretization,
            pressure_system,
        ),
        discretization_controls=discretization,
        iteration_controls=iteration,
        linear_solvers=build_collocated_ns_linear_solvers(),
    )
    return solver, velocity, viscosity * rate


def _exp0015_solver(n=8):
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
        NSFVMSimpleModel,
        steady_traction_mms_simple_profile,
    )
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(15)

    def boundary_factory(mesh, pde):
        return EngineeringBoundaryConditions(
            mesh,
            patches=[
                BoundaryPatch("fixed", pde.is_velocity_boundary),
                BoundaryPatch("outlet", pde.is_outlet_boundary),
            ],
            conditions=[
                BoundaryCondition(
                    "velocity",
                    "fixed",
                    "dirichlet",
                    pde.dirichlet_velocity,
                ),
                BoundaryCondition("velocity", "outlet", "natural", None),
                BoundaryCondition(
                    "momentum",
                    "outlet",
                    "traction",
                    pde.traction,
                ),
            ],
        )

    profile = steady_traction_mms_simple_profile()
    return NSFVMSimpleModel(
        {
            "pde": pde,
            "mesh_type": "uniform_quad",
            "nx": n,
            "ny": n,
            "profile": profile,
            "boundary_conditions": boundary_factory,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )


def _smooth_perturbed_triangle_mesh(n, amplitude=0.18):
    from fealpy.mesh import TriangleMesh

    regular = TriangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=n,
        ny=n,
    )
    node = bm.copy(regular.entity("node"))
    x = node[:, 0]
    y = node[:, 1]
    eps = 1.0e-12
    interior = (
        (x > eps)
        & (x < 1.0 - eps)
        & (y > eps)
        & (y < 1.0 - eps)
    )
    h = 1.0 / n
    dx = amplitude * h * bm.sin(2.0 * bm.pi * x) * bm.sin(bm.pi * y)
    dy = -amplitude * h * bm.sin(bm.pi * x) * bm.sin(2.0 * bm.pi * y)
    node = bm.set_at(node, (interior, 0), x[interior] + dx[interior])
    node = bm.set_at(node, (interior, 1), y[interior] + dy[interior])
    return TriangleMesh(node, bm.copy(regular.entity("cell")))


def _exp0015_direct_solver(mesh, pde):
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        CollocatedPressureSystemControls,
        CollocatedSimpleSolver,
        EngineeringBoundaryConditions,
        SimpleDiscretizationControls,
        SimpleIterationControls,
        build_collocated_ns_linear_solvers,
        resolve_simple_boundary_conditions,
    )

    boundary = EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch("fixed", pde.is_velocity_boundary),
            BoundaryPatch("outlet", pde.is_outlet_boundary),
        ],
        conditions=[
            BoundaryCondition(
                "velocity",
                "fixed",
                "dirichlet",
                pde.dirichlet_velocity,
            ),
            BoundaryCondition("velocity", "outlet", "natural", None),
            BoundaryCondition(
                "momentum",
                "outlet",
                "traction",
                pde.traction,
            ),
        ],
    )
    discretization = SimpleDiscretizationControls(
        pressure_gradient_method="layered_lsq",
        velocity_gradient_method="layered_lsq",
        gradient_layer_weights=(1.0, 0.25),
        gradient_boundary_weight=1.0,
        diffusion_method="over_relaxed",
        momentum_face_interpolation="average",
        pressure_response_interpolation="average",
        rhie_chow_velocity_interpolation="average",
        spatial_face_velocity_scheme="second_order_reconstructed",
        face_flux_correction_scheme="cell_anchored_quadratic",
        face_flux_quadrature_order=5,
    )
    iteration = SimpleIterationControls(
        max_iterations=1000,
        momentum_equation_relaxation=0.9,
        momentum_relative_tolerance=1.0e-7,
        mass_relative_tolerance=1.0e-8,
        momentum_nonorthogonal_max_iterations=50,
        pressure_nonorthogonal_max_iterations=50,
    )
    pressure_system = CollocatedPressureSystemControls(
        pure_neumann_closure=PressureClosureKind.GAUGE,
    )
    return CollocatedSimpleSolver(
        diffusion_coef=float(pde.mu),
        convection_coef=float(pde.rho),
        source=pde.source,
        boundary_conditions=resolve_simple_boundary_conditions(
            mesh,
            boundary,
            discretization,
            pressure_system,
        ),
        discretization_controls=discretization,
        iteration_controls=iteration,
        linear_solvers=build_collocated_ns_linear_solvers(),
    )


def _exp0015_error_row(mesh_family, n):
    from fealpy.fvm import FVMGeometry
    from fealpy.model.navier_stokes.exp0015 import Exp0015

    pde = Exp0015()
    if mesh_family == "quad":
        mesh = pde.init_mesh["uniform_quad"](nx=n, ny=n)
    elif mesh_family == "tri_smooth_perturbed":
        mesh = _smooth_perturbed_triangle_mesh(n)
    else:
        raise ValueError(mesh_family)
    solver = _exp0015_direct_solver(mesh, pde)
    solve_result = solver.solve()
    geometry = FVMGeometry(mesh)
    exact_velocity = geometry.cell_integral(
        lambda points, _: pde.velocity(points),
        q=7,
    ) / geometry.cell_measure[:, None]
    exact_pressure = geometry.cell_integral(
        lambda points, _: pde.pressure(points),
        q=7,
    ) / geometry.cell_measure
    exact_face_velocity = geometry.face_integral(
        lambda points, _: pde.velocity(points),
        q=7,
    ) / geometry.face_measure[:, None]
    exact_face_flux = bm.einsum(
        "fi,fi->f",
        exact_face_velocity,
        geometry.S_f,
    )
    total_measure = bm.sum(geometry.cell_measure)
    velocity_difference = solve_result.velocity - exact_velocity
    pressure_difference = solve_result.pressure - exact_pressure
    velocity_error = bm.sqrt(
        bm.sum(
            geometry.cell_measure
            * bm.sum(velocity_difference**2, axis=1)
        )
        / total_measure
    )
    pressure_error = bm.sqrt(
        bm.sum(geometry.cell_measure * pressure_difference**2)
        / total_measure
    )
    flux_density_difference = (
        solve_result.face_flux - exact_face_flux
    ) / geometry.face_measure
    face_flux_error = bm.sqrt(
        bm.sum(
            geometry.face_measure * flux_density_difference**2
        )
        / bm.sum(geometry.face_measure)
    )
    residual = solve_result.residual_history[-1]
    diagnostics = (
        solver.spatial_face_velocity.face_flux_reconstruct.diagnostics()
    )
    return {
        "mesh_family": mesh_family,
        "n": n,
        "h": float(
            bm.to_numpy(
                bm.sqrt(bm.sum(geometry.cell_measure) / solver.discretization.NC)
            )
        ),
        "converged": solve_result.converged,
        "momentum_residual_relative": residual.momentum_relative_l2,
        "mass_residual": residual.mass_relative_l2,
        "velocity_error": float(bm.to_numpy(velocity_error)),
        "pressure_error": float(bm.to_numpy(pressure_error)),
        "face_flux_error": float(bm.to_numpy(face_flux_error)),
        "wide_stencil_fallbacks": diagnostics["fallback_cell_count"],
    }


def _exp0015_traction_convergence_matrix():
    rows = [
        _exp0015_error_row(mesh_family, n)
        for mesh_family in ("quad", "tri_smooth_perturbed")
        for n in (8, 16, 32)
    ]
    for mesh_family in ("quad", "tri_smooth_perturbed"):
        family_rows = [
            row for row in rows
            if row["mesh_family"] == mesh_family
        ]
        for coarse, fine in zip(family_rows[:-1], family_rows[1:]):
            denominator = np.log(coarse["h"] / fine["h"])
            for error, order in (
                ("velocity_error", "velocity_order"),
                ("pressure_error", "pressure_order"),
                ("face_flux_error", "face_flux_order"),
            ):
                fine[order] = (
                    np.log(coarse[error] / fine[error])
                    / denominator
                )
    return rows


def test_constant_pressure_force_removes_traction_outlet_face_flux():
    bm.set_backend("numpy")
    solver = _operator_solver(traction=(0.0, 0.0))
    pressure_value = 2.5
    pressure = pressure_value * bm.ones(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    pressure_gradient = bm.zeros((solver.discretization.NC, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    faces = _resolved_traction_faces(solver)
    owner = solver.discretization.geometry.owner[faces]
    expected_cell = bm.zeros((solver.discretization.NC, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    expected_cell = bm.index_add(
        expected_cell,
        owner,
        -pressure_value * solver.discretization.geometry.S_f[faces],
        axis=0,
    )

    actual = solver.momentum.spatial_operator.pressure_source(
        pressure,
        pressure_gradient=pressure_gradient,
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(solver.discretization.cell_vector_to_dofs(expected_cell)),
        atol=1.0e-14,
    )


def test_pressure_force_without_traction_equals_gradient_source():
    bm.set_backend("numpy")
    solver = _operator_solver()
    centers = solver.discretization.geometry.cell_center
    pressure = 0.7 + 0.3 * centers[:, 0] - 0.2 * centers[:, 1]
    pressure_gradient = solver.pressure_gradient.cell_gradient(pressure)

    actual = solver.momentum.spatial_operator.pressure_source(
        pressure,
        pressure_gradient=pressure_gradient,
    )
    expected = solver.momentum.spatial_operator.pressure_source(
        pressure,
        pressure_gradient=pressure_gradient,
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        atol=1.0e-14,
    )


def test_prescribed_traction_rhs_is_owner_oriented_face_integral():
    bm.set_backend("numpy")
    solver = _operator_solver(traction=(2.0, -3.0))
    faces = _resolved_traction_faces(solver)
    values = bm.broadcast_to(
        bm.asarray([2.0, -3.0], dtype=solver.discretization.geometry.cell_measure.dtype),
        (faces.shape[0], solver.discretization.GD),
    )
    expected_cell = bm.zeros((solver.discretization.NC, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    expected_cell = bm.index_add(
        expected_cell,
        solver.discretization.geometry.owner[faces],
        solver.discretization.geometry.mag_S_f[faces, None] * values,
        axis=0,
    )

    actual = solver.momentum.spatial_operator.boundary_source(
        solver.discretization.cell_vector_to_dofs(expected_cell)
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(solver.discretization.cell_vector_to_dofs(expected_cell)),
        atol=1.0e-14,
    )


def test_simple_traction_uses_resolved_momentum_operator():
    bm.set_backend("numpy")
    solver = _operator_solver(traction=(0.0, 0.0))

    assert solver.momentum.spatial_operator.momentum_boundary.traction is not None
    assert _resolved_traction_faces(solver).shape[0] > 0


def test_traction_outlet_separates_state_and_correction_pressure_boundaries():
    bm.set_backend("numpy")

    solver = _operator_solver(traction=(0.0, 0.0))

    assert (
        solver.pressure_gradient.boundary.dirichlet_faces.shape[0]
        == 0
    )
    assert (
        solver.pressure_system.boundary.closure
        is PressureClosureKind.DIRICHLET
    )
    assert solver.pressure_gradient.boundary.dirichlet_faces.shape[0] == 0
    correction_gradient = solver.pressure_system.boundary.gradient
    assert correction_gradient is not solver.pressure_gradient
    assert correction_gradient.boundary.dirichlet_faces.shape[0] > 0


def test_state_rhie_chow_gradient_difference_is_zero_on_traction_outlet():
    bm.set_backend("numpy")
    solver = _operator_solver(traction=(0.0, 0.0))
    centers = solver.discretization.geometry.cell_center
    pressure = centers[:, 0] ** 2 + 0.3 * centers[:, 0] * centers[:, 1]

    gradient_difference = solver.rhie_chow.pressure_gradient_difference(
        pressure,
        solver.pressure_gradient.cell_gradient(pressure),
    )
    faces = _resolved_traction_faces(solver)

    np.testing.assert_allclose(
        bm.to_numpy(gradient_difference[faces]),
        0.0,
        atol=1.0e-14,
    )


def test_pressure_correction_keeps_nonzero_traction_outlet_normal_response():
    bm.set_backend("numpy")
    solver = _operator_solver(traction=(0.0, 0.0))
    pressure_correction = bm.ones(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    response_coef = bm.ones(solver.discretization.NF, dtype=solver.discretization.geometry.cell_measure.dtype)

    flux = solver.pressure_system.correction_flux(
        pressure_correction,
        response_coef,
        pressure_gradient=(
            solver.pressure_gradient.cell_gradient(
                pressure_correction
            )
        ),
    )
    faces = _resolved_traction_faces(solver)

    assert (
        float(
            bm.max(
                bm.abs(flux[faces])
            )
        )
        > 1.0e-12
    )


def test_traction_outlet_only_masks_its_momentum_cross_diffusion_flux(
    monkeypatch,
):
    bm.set_backend("numpy")
    import fealpy.fvm.collocated_momentum_equation as components

    solver = _operator_solver(traction=(0.0, 0.0))
    face_flux = bm.reshape(
        bm.arange(solver.discretization.NF * solver.discretization.GD, dtype=solver.discretization.geometry.cell_measure.dtype) + 1.0,
        (solver.discretization.NF, solver.discretization.GD),
    )

    monkeypatch.setattr(
        solver.momentum.spatial_operator,
        "boundary_corrected_face_gradient",
        lambda velocity, interpolation_method: bm.zeros(
            (solver.discretization.NF, solver.discretization.GD, solver.discretization.GD),
            dtype=solver.discretization.geometry.cell_measure.dtype,
        ),
    )
    monkeypatch.setattr(
        components,
        "scalar_cross_diffusion_face_flux",
        lambda *args, **kwargs: bm.copy(face_flux),
        raising=False,
    )

    actual = solver.momentum.spatial_operator.nonorthogonal_rhs(
        bm.zeros((solver.discretization.NC, solver.discretization.GD), dtype=solver.discretization.geometry.cell_measure.dtype)
    )
    faces = _resolved_traction_faces(solver)
    expected_face_flux = bm.set_at(
        bm.copy(face_flux),
        faces,
        bm.zeros_like(face_flux[faces]),
    )
    expected = solver.discretization.cell_vector_to_dofs(
        solver.discretization.geometry.scatter_face_flux_to_cells(expected_face_flux)
    )

    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        atol=1.0e-14,
    )


def test_traction_linear_probe_has_machine_precision_momentum_balance():
    bm.set_backend("numpy")
    from fealpy.fvm.solver_diagnostics import normalized_equation_residual

    solver, velocity_function, pressure_value = _linear_probe_solver(
        traction=True
    )
    velocity = velocity_function(solver.discretization.geometry.cell_center)
    face_velocity = velocity_function(solver.discretization.geometry.face_center)
    pressure = pressure_value * bm.ones(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    pressure_gradient = bm.zeros(
        (solver.discretization.NC, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )

    balance = solver.momentum.balance(
        pressure,
        velocity,
        face_velocity,
        pressure_gradient=pressure_gradient,
    )
    metrics = normalized_equation_residual(
        balance.lhs,
        balance.rhs,
    )

    assert metrics.relative < 1.0e-11


def test_traction_pressure_shift_changes_residual_by_outlet_pressure_force():
    bm.set_backend("numpy")
    solver, velocity_function, pressure_value = _linear_probe_solver(
        traction=True
    )
    velocity = velocity_function(solver.discretization.geometry.cell_center)
    face_velocity = velocity_function(solver.discretization.geometry.face_center)
    pressure_gradient = bm.zeros(
        (solver.discretization.NC, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    pressure = pressure_value * bm.ones(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    shift = 0.37

    base = solver.momentum.balance(
        pressure,
        velocity,
        face_velocity,
        pressure_gradient=pressure_gradient,
    ).residual
    shifted = solver.momentum.balance(
        pressure + shift,
        velocity,
        face_velocity,
        pressure_gradient=pressure_gradient,
    ).residual
    expected_cell = bm.zeros(
        (solver.discretization.NC, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    expected_cell = bm.index_add(
        expected_cell,
        solver.discretization.geometry.owner[_resolved_traction_faces(solver)],
        -shift * solver.discretization.geometry.S_f[_resolved_traction_faces(solver)],
        axis=0,
    )

    np.testing.assert_allclose(
        bm.to_numpy(shifted - base),
        bm.to_numpy(solver.discretization.cell_vector_to_dofs(expected_cell)),
        atol=1.0e-13,
    )


def test_pressure_outlet_linear_probe_uses_gradient_pressure_source():
    bm.set_backend("numpy")
    solver, velocity_function, pressure_value = _linear_probe_solver(
        traction=False
    )
    velocity = velocity_function(solver.discretization.geometry.cell_center)
    face_velocity = velocity_function(solver.discretization.geometry.face_center)
    pressure = pressure_value * bm.ones(solver.discretization.NC, dtype=solver.discretization.geometry.cell_measure.dtype)
    pressure_gradient = bm.zeros(
        (solver.discretization.NC, solver.discretization.GD),
        dtype=solver.discretization.geometry.cell_measure.dtype,
    )
    actual = solver.momentum.spatial_operator.pressure_source(
        pressure,
        pressure_gradient=pressure_gradient,
    )
    expected = solver.discretization.cell_vector_to_dofs(
        solver.discretization.geometry.cell_measure[:, None] * pressure_gradient
    )

    assert _resolved_traction_faces(solver).shape[0] == 0
    np.testing.assert_allclose(
        bm.to_numpy(actual),
        bm.to_numpy(expected),
        atol=1.0e-14,
    )


def test_exp0015_single_grid_traction_simple_converges():
    bm.set_backend("numpy")
    solver = _exp0015_solver(n=8)

    result = solver.solve()
    pressure_mean = (
        bm.sum(result.pressure * solver.solver.discretization.geometry.cell_measure)
        / bm.sum(solver.solver.discretization.geometry.cell_measure)
    )

    assert result.converged is True
    assert result.termination_reason == "fixed_point_residuals"
    assert result.residual_history[-1].momentum_relative_l2 <= 1.0e-7
    assert result.residual_history[-1].mass_relative_l2 <= 1.0e-8
    assert bool(bm.to_numpy(bm.all(bm.isfinite(result.velocity))))
    assert bool(bm.to_numpy(bm.all(bm.isfinite(result.pressure))))
    assert bool(bm.to_numpy(bm.all(bm.isfinite(result.face_flux))))
    assert abs(float(bm.to_numpy(pressure_mean))) > 1.0e-6


def test_exp0015_traction_matrix_meets_effective_orders():
    rows = _exp0015_traction_convergence_matrix()
    assert len(rows) == 6
    assert all(row["converged"] for row in rows)
    assert all(
        row["momentum_residual_relative"] <= 1.0e-7
        and row["mass_residual"] <= 1.0e-8
        for row in rows
    )
    assert all(row["wide_stencil_fallbacks"] == 0 for row in rows)
    final = {
        row["mesh_family"]: row
        for row in rows
        if row["n"] == 32
    }

    assert set(final) == {"quad", "tri_smooth_perturbed"}
    for family in final:
        family_rows = [
            row for row in rows
            if row["mesh_family"] == family
        ]
        for error in ("velocity_error", "pressure_error", "face_flux_error"):
            values = [row[error] for row in family_rows]
            assert values[0] > values[1] > values[2]
    for row in final.values():
        assert row["velocity_order"] >= 1.5
        assert row["pressure_order"] >= 1.0
        assert row["face_flux_order"] >= 1.5

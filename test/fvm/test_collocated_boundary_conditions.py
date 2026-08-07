from dataclasses import fields
import numpy as np
import pytest


def _resolve_simple(mesh, boundary_conditions):
    from fealpy.fvm import (
        CollocatedPressureSystemControls,
        SimpleDiscretizationControls,
        resolve_simple_boundary_conditions,
    )

    return resolve_simple_boundary_conditions(
        mesh,
        boundary_conditions,
        SimpleDiscretizationControls(),
        CollocatedPressureSystemControls(),
    )


def test_resolved_boundary_types_are_frozen_typed_data_contracts():
    from fealpy.fvm.collocated_boundary_conditions import (
        ResolvedMomentumBoundary,
        ResolvedPhysicalBoundaryConditions,
        ResolvedPisoBoundaryConditions,
        ResolvedPressureSystemBoundary,
        ResolvedPressureStateBoundary,
        ResolvedRhieChowBoundary,
        ResolvedSimpleBoundaryConditions,
        ResolvedVelocityBoundary,
    )

    resolved_types = (
        ResolvedVelocityBoundary,
        ResolvedPressureStateBoundary,
        ResolvedMomentumBoundary,
        ResolvedPressureSystemBoundary,
        ResolvedRhieChowBoundary,
        ResolvedPhysicalBoundaryConditions,
        ResolvedSimpleBoundaryConditions,
        ResolvedPisoBoundaryConditions,
    )
    assert all(item.__dataclass_params__.frozen for item in resolved_types)
    assert all(
        field.type not in {object, "object"}
        for item in resolved_types
        for field in fields(item)
    )


def test_resolved_boundary_data_has_one_authoritative_owner():
    from fealpy.fvm.collocated_boundary_conditions import (
        ResolvedMomentumBoundary,
        ResolvedPressureSystemBoundary,
        ResolvedPressureStateBoundary,
        ResolvedRhieChowBoundary,
        ResolvedVelocityBoundary,
    )

    assert {field.name for field in fields(ResolvedVelocityBoundary)} == {
        "dirichlet_face_values",
        "neumann_faces",
        "neumann_sn_grad",
        "natural_faces",
        "dirichlet_operator",
        "gradient",
    }
    assert {field.name for field in fields(ResolvedPressureStateBoundary)} == {
        "gradient",
    }
    assert {field.name for field in fields(ResolvedMomentumBoundary)} == {
        "velocity",
        "traction",
    }
    assert {
        field.name for field in fields(ResolvedPressureSystemBoundary)
    } == {
        "dirichlet_operator",
        "gradient",
        "closure",
    }
    assert {field.name for field in fields(ResolvedRhieChowBoundary)} == {
        "pressure_state",
        "zero_gradient_difference_faces",
    }


def test_normalized_pde_boundary_uses_typed_variable_data():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=1,
        ny=1,
    )
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: bm.zeros_like(points),
    )

    assert boundary.velocity.dirichlet_faces.shape == (4,)
    assert boundary.velocity.neumann_faces.shape == (0,)
    assert boundary.velocity.natural_faces.shape == (0,)
    assert boundary.pressure.dirichlet_faces.shape == (0,)
    assert boundary.momentum.traction_faces.shape == (0,)


def test_simple_resolver_normalizes_engineering_and_pde_boundaries_equally():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)

    def fixed_selector(points):
        return bm.abs(points[..., 0] - 1.0) > 1.0e-12

    def outlet_selector(points):
        return bm.abs(points[..., 0] - 1.0) < 1.0e-12

    def velocity(points):
        return bm.broadcast_to(
            bm.asarray([1.0, 0.0], dtype=points.dtype),
            points.shape,
        )

    engineering = EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch("fixed", fixed_selector),
            BoundaryPatch("outlet", outlet_selector),
        ],
        conditions=[
            BoundaryCondition("velocity", "fixed", "dirichlet", velocity),
            BoundaryCondition("velocity", "outlet", "natural", None),
        ],
    )
    pde_boundary = engineering.to_pde_boundary()
    resolved_engineering = _resolve_simple(mesh, engineering)
    resolved_pde = _resolve_simple(mesh, pde_boundary)

    assert resolved_engineering.physical.geometry is resolved_pde.physical.geometry
    np.testing.assert_array_equal(
        resolved_engineering.physical.velocity.dirichlet_operator.faces,
        resolved_pde.physical.velocity.dirichlet_operator.faces,
    )


def test_piso_resolver_rejects_momentum_traction():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        PDEBoundaryConditions,
        CollocatedPressureSystemControls,
        PisoSolverControls,
        resolve_piso_boundary_conditions,
    )
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)

    def velocity(points):
        return bm.zeros_like(points)

    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=velocity,
        momentum_traction=lambda points: bm.zeros_like(points),
        momentum_traction_selector=(
            lambda points: bm.abs(points[..., 0] - 1.0) < 1.0e-12
        ),
    )

    with pytest.raises(
        ValueError,
        match="momentum traction is not supported.*PISO",
    ):
        resolve_piso_boundary_conditions(
            mesh,
            boundary,
            PisoSolverControls(),
            CollocatedPressureSystemControls(),
        )


def test_boundary_resolver_rejects_a_different_mesh():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        PDEBoundaryConditions,
    )
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    boundary_mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    solver_mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    boundary = PDEBoundaryConditions(
        boundary_mesh,
        dirichlet_velocity=lambda points: bm.zeros_like(points),
    )

    with pytest.raises(
        ValueError,
        match="mesh-bound boundary conditions require their original mesh",
    ):
        _resolve_simple(solver_mesh, boundary)


def test_homogeneous_traction_builds_active_momentum_operator():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        PDEBoundaryConditions,
    )
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    outlet = lambda points: bm.abs(points[..., 0] - 1.0) < 1.0e-12
    fixed = lambda points: ~outlet(points)
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: bm.zeros_like(points),
        dirichlet_velocity_selector=fixed,
        natural_velocity_selector=outlet,
        momentum_traction=lambda points: bm.zeros_like(points),
        momentum_traction_selector=outlet,
    )

    resolved = _resolve_simple(mesh, boundary)
    operator = resolved.physical.momentum

    assert operator.traction is not None
    assert operator.traction.faces.shape[0] > 0


def _resolved_traction_case(traction=(2.0, -3.0)):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        PDEBoundaryConditions,
        TractionBC,
    )
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    outlet = lambda points: bm.abs(points[..., 0] - 1.0) < 1.0e-12
    fixed = lambda points: ~outlet(points)

    def traction_value(points):
        return bm.broadcast_to(
            bm.asarray(traction, dtype=points.dtype),
            points.shape,
        )

    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: bm.zeros_like(points),
        dirichlet_velocity_selector=fixed,
        momentum_traction=traction_value,
        momentum_traction_selector=outlet,
    )
    resolved = _resolve_simple(mesh, boundary)
    traction_operator = resolved.physical.momentum.traction
    direct = TractionBC(
        geometry=traction_operator.geometry,
        faces=traction_operator.faces,
        face_average_values=traction_operator.face_average_values,
        cell_source=traction_operator.cell_source,
    )
    return resolved, direct


def test_resolved_traction_source_and_pressure_force_match_traction_bc():
    from fealpy.backend import backend_manager as bm

    bm.set_backend("numpy")
    resolved, direct = _resolved_traction_case()
    geometry = resolved.physical.geometry
    pressure = 2.5 * bm.ones(geometry.NC, dtype=geometry.cell_center.dtype)
    pressure_gradient = bm.zeros(
        (geometry.NC, geometry.GD),
        dtype=geometry.cell_center.dtype,
    )
    cell_force = bm.zeros_like(pressure_gradient)

    np.testing.assert_allclose(
        resolved.physical.momentum.source(cell_force),
        direct.source(cell_force),
    )
    np.testing.assert_allclose(
        resolved.physical.momentum.apply_pressure_force(
            cell_force,
            pressure,
            pressure_gradient,
        ),
        direct.apply_pressure_force(
            cell_force,
            pressure,
            pressure_gradient,
        ),
    )


def test_resolved_traction_convection_and_cross_flux_match_traction_bc():
    from fealpy.backend import backend_manager as bm

    bm.set_backend("numpy")
    resolved, direct = _resolved_traction_case(traction=(0.0, 0.0))
    geometry = resolved.physical.geometry
    faces = direct.faces
    owner = geometry.owner[faces]
    cell_velocity = bm.copy(geometry.cell_center)
    reconstructed = bm.zeros_like(geometry.face_center)
    reconstructed = bm.set_at(
        reconstructed,
        faces,
        cell_velocity[owner]
        + bm.broadcast_to(
            bm.asarray([0.4, -0.2], dtype=cell_velocity.dtype),
            (faces.shape[0], geometry.GD),
        ),
    )
    convection = bm.zeros_like(geometry.face_center)
    convection = bm.set_at(
        convection,
        faces,
        bm.broadcast_to(
            bm.asarray([1.5, 0.0], dtype=cell_velocity.dtype),
            (faces.shape[0], geometry.GD),
        ),
    )
    natural_diagonal = (
        resolved.physical.momentum.natural_convection_diagonal(
            convection
        )
    )
    actual_source = resolved.physical.momentum.convection_source(
        convection,
        cell_velocity,
        reconstructed,
    )
    direct_source = direct.convection_source(
        convection,
        cell_velocity,
        reconstructed,
    )
    np.testing.assert_allclose(
        natural_diagonal,
        bm.zeros(geometry.NC, dtype=cell_velocity.dtype),
    )
    np.testing.assert_allclose(actual_source, direct_source)

    face_flux = bm.reshape(
        bm.arange(
            geometry.NF * geometry.GD,
            dtype=cell_velocity.dtype,
        )
        + 1.0,
        (geometry.NF, geometry.GD),
    )
    np.testing.assert_allclose(
        resolved.physical.momentum.apply_cross_diffusion_flux(face_flux),
        direct.apply_cross_diffusion_flux(face_flux),
    )


def _resolved_mixed_velocity_case():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        PDEBoundaryConditions,
    )
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    left = lambda points: bm.abs(points[..., 0]) < 1.0e-12
    top = lambda points: bm.abs(points[..., 1] - 1.0) < 1.0e-12
    right = lambda points: bm.abs(points[..., 0] - 1.0) < 1.0e-12

    def velocity(points):
        return bm.stack(
            [
                points[..., 1] ** 2,
                2.0 * bm.ones(points.shape[:-1]),
            ],
            axis=-1,
        )

    def sn_grad(points):
        return bm.broadcast_to(
            bm.asarray([1.0, -2.0], dtype=points.dtype),
            points.shape,
        )

    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=velocity,
        dirichlet_velocity_selector=left,
        neumann_velocity=sn_grad,
        neumann_velocity_selector=top,
        natural_velocity_selector=right,
    )
    return _resolve_simple(mesh, boundary)


def test_resolved_velocity_diffusion_combines_dirichlet_and_neumann_terms():
    from fealpy.backend import backend_manager as bm
    from fealpy.sparse import spdiags

    bm.set_backend("numpy")
    resolved = _resolved_mixed_velocity_case()
    physical = resolved.physical
    geometry = physical.geometry
    momentum = physical.momentum
    coefficient = 2.0
    matrix = spdiags(
        bm.zeros(geometry.NC, dtype=geometry.cell_center.dtype),
        0,
        geometry.NC,
        geometry.NC,
    )
    rhs = bm.zeros(
        geometry.NC * geometry.GD,
        dtype=geometry.cell_center.dtype,
    )

    actual_matrix, actual_rhs = momentum.apply_diffusion(
        matrix,
        rhs,
        coefficient,
    )
    actual_neumann_source = momentum.diffusion_source(
        coefficient,
        rhs,
    )

    dirichlet_faces = physical.velocity.dirichlet_operator.faces
    dirichlet_values = physical.velocity.dirichlet_operator.values
    factor = geometry.diffusion_face_decomposition(
        "over_relaxed"
    ).orthogonal_factor[dirichlet_faces]
    dirichlet_weight = coefficient * factor
    expected_diagonal = bm.index_add(
        bm.zeros(geometry.NC, dtype=rhs.dtype),
        geometry.owner[dirichlet_faces],
        dirichlet_weight,
        axis=0,
    )
    expected_cell_rhs = bm.index_add(
        bm.zeros((geometry.NC, geometry.GD), dtype=rhs.dtype),
        geometry.owner[dirichlet_faces],
        dirichlet_weight[:, None] * dirichlet_values,
        axis=0,
    )
    neumann_faces = physical.velocity.neumann_faces
    neumann_values = physical.velocity.neumann_sn_grad
    expected_neumann_source = bm.index_add(
        bm.zeros_like(expected_cell_rhs),
        geometry.owner[neumann_faces],
        (
            coefficient
            * geometry.mag_S_f[neumann_faces, None]
            * neumann_values
        ),
        axis=0,
    )
    expected_dirichlet_rhs = bm.reshape(
        bm.swapaxes(expected_cell_rhs, 0, 1),
        (-1,),
    )

    np.testing.assert_allclose(
        np.diag(bm.to_numpy(actual_matrix.to_dense())),
        bm.to_numpy(expected_diagonal),
    )
    np.testing.assert_allclose(actual_rhs, expected_dirichlet_rhs)
    np.testing.assert_allclose(
        actual_neumann_source,
        expected_neumann_source,
    )


def test_resolved_momentum_returns_only_natural_convection_cell_diagonal():
    from fealpy.backend import backend_manager as bm

    bm.set_backend("numpy")
    resolved = _resolved_mixed_velocity_case()
    physical = resolved.physical
    geometry = physical.geometry
    convection = bm.zeros_like(geometry.face_center)
    boundary_faces = bm.nonzero(geometry.is_boundary)[0]
    convection = bm.set_at(
        convection,
        boundary_faces,
        bm.broadcast_to(
            bm.asarray([1.0, 0.0], dtype=convection.dtype),
            (boundary_faces.shape[0], geometry.GD),
        ),
    )

    actual_diagonal = physical.momentum.natural_convection_diagonal(
        convection,
    )

    natural_faces = physical.velocity.natural_faces
    natural_flux = bm.einsum(
        "ij,ij->i",
        convection[natural_faces],
        geometry.S_f[natural_faces],
    )
    expected_diagonal = bm.index_add(
        bm.zeros(geometry.NC, dtype=convection.dtype),
        geometry.owner[natural_faces],
        natural_flux,
        axis=0,
    )

    np.testing.assert_allclose(
        actual_diagonal,
        expected_diagonal,
    )


def test_resolved_velocity_stores_selected_face_values():
    from fealpy.backend import backend_manager as bm

    bm.set_backend("numpy")
    resolved = _resolved_mixed_velocity_case()
    velocity = resolved.physical.velocity
    center_faces = velocity.dirichlet_operator.faces
    center_values = velocity.dirichlet_operator.values
    average_faces = velocity.dirichlet_operator.faces
    average_values = velocity.dirichlet_face_values
    centers = resolved.physical.geometry.face_center[center_faces]
    order = bm.argsort(centers[:, 1])

    np.testing.assert_array_equal(center_faces, average_faces)
    np.testing.assert_allclose(
        center_values[order, 0],
        [1.0 / 16.0, 9.0 / 16.0],
    )
    np.testing.assert_allclose(
        average_values[order, 0],
        [1.0 / 12.0, 7.0 / 12.0],
    )
    np.testing.assert_allclose(average_values[:, 1], 2.0)


def test_velocity_face_values_only_integrate_for_active_flux_correction(
    monkeypatch,
):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        CollocatedPressureSystemControls,
        PDEBoundaryConditions,
        PisoSolverControls,
        SimpleDiscretizationControls,
        resolve_piso_boundary_conditions,
        resolve_simple_boundary_conditions,
    )
    from fealpy.fvm.fvm_geometry import FVMGeometry
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    calls = []
    original = FVMGeometry.face_integral

    def record_face_integral(geometry, integrand, *, q=3):
        calls.append(q)
        return original(geometry, integrand, q=q)

    monkeypatch.setattr(
        FVMGeometry,
        "face_integral",
        record_face_integral,
    )

    def boundary(mesh):
        return PDEBoundaryConditions(
            mesh,
            dirichlet_velocity=lambda points: points**2,
        )

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    no_flux_controls = SimpleDiscretizationControls(
        face_flux_correction_scheme="none",
    )
    simple_center = resolve_simple_boundary_conditions(
        mesh,
        boundary(mesh),
        no_flux_controls,
        CollocatedPressureSystemControls(),
    )
    assert calls == []
    np.testing.assert_allclose(
        simple_center.physical.velocity.dirichlet_face_values,
        simple_center.physical.velocity.dirichlet_operator.values,
    )

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    piso = resolve_piso_boundary_conditions(
        mesh,
        boundary(mesh),
        PisoSolverControls(),
        CollocatedPressureSystemControls(),
    )
    assert calls == []
    np.testing.assert_allclose(
        piso.physical.velocity.dirichlet_face_values,
        piso.physical.velocity.dirichlet_operator.values,
    )

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    average_controls = SimpleDiscretizationControls(
        face_flux_correction_scheme="cell_anchored_quadratic",
        face_flux_quadrature_order=5,
    )
    resolve_simple_boundary_conditions(
        mesh,
        boundary(mesh),
        average_controls,
        CollocatedPressureSystemControls(),
    )
    assert calls == [5]


def test_vector_face_average_preserves_entity_quadrature_axes():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import FVMGeometry
    from fealpy.fvm.collocated_boundary_conditions import (
        vector_face_average,
    )
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    geometry = FVMGeometry(mesh)
    observed_shapes = []

    def value(points):
        observed_shapes.append(points.shape)
        return bm.stack(
            (points[..., 0] ** 2, 1.0 + points[..., 1]),
            axis=-1,
        )

    average = vector_face_average(
        geometry,
        geometry.boundary_faces,
        value,
        quadrature_order=3,
    )

    assert observed_shapes
    assert all(len(shape) == 3 for shape in observed_shapes)
    assert average.shape == (
        geometry.boundary_faces.shape[0],
        geometry.GD,
    )


@pytest.mark.parametrize(
    ("with_pressure", "with_traction"),
    [(True, False), (False, True), (True, True)],
)
def test_simple_pressure_correction_fixes_pressure_and_traction_faces(
    with_pressure,
    with_traction,
):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.fvm import PressureClosureKind
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    left = lambda points: bm.abs(points[..., 0]) < 1.0e-12
    right = lambda points: bm.abs(points[..., 0] - 1.0) < 1.0e-12
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: bm.zeros_like(points),
        dirichlet_pressure=(
            (lambda points: bm.zeros(points.shape[:-1], dtype=points.dtype))
            if with_pressure
            else None
        ),
        dirichlet_pressure_selector=left if with_pressure else None,
        momentum_traction=(
            (lambda points: bm.zeros_like(points))
            if with_traction
            else None
        ),
        momentum_traction_selector=right if with_traction else None,
    )
    resolved = _resolve_simple(mesh, boundary)
    policy = resolved.pressure_correction
    assert set(vars(policy)) == {
        "dirichlet_operator",
        "gradient",
        "closure",
    }
    expected_parts = []
    if with_pressure:
        expected_parts.append(
            resolved.physical.pressure_state.gradient.boundary.dirichlet_faces
        )
    if with_traction:
        expected_parts.append(resolved.physical.momentum.traction.faces)
    expected = np.unique(
        np.concatenate([bm.to_numpy(part) for part in expected_parts])
    )

    assert policy.closure is PressureClosureKind.DIRICHLET
    np.testing.assert_array_equal(
        bm.to_numpy(policy.dirichlet_operator.faces),
        expected,
    )
    np.testing.assert_array_equal(
        bm.to_numpy(policy.dirichlet_operator.values),
        np.zeros(expected.shape[0]),
    )


def test_simple_rhie_chow_policy_resolves_pressure_and_traction_constraints():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        PDEBoundaryConditions,
    )
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    left = lambda points: bm.abs(points[..., 0]) < 1.0e-12
    right = lambda points: bm.abs(points[..., 0] - 1.0) < 1.0e-12
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: bm.zeros_like(points),
        dirichlet_pressure=lambda points: bm.ones(
            points.shape[:-1],
            dtype=points.dtype,
        ),
        dirichlet_pressure_selector=left,
        momentum_traction=lambda points: bm.zeros_like(points),
        momentum_traction_selector=right,
    )
    resolved = _resolve_simple(mesh, boundary)
    policy = resolved.rhie_chow

    np.testing.assert_array_equal(
        policy.zero_gradient_difference_faces,
        resolved.physical.momentum.traction.faces,
    )
    assert policy.pressure_state is resolved.physical.pressure_state
    np.testing.assert_array_equal(
        policy.pressure_state.gradient.boundary.dirichlet_faces,
        resolved.physical.pressure_state.gradient.boundary.dirichlet_faces,
    )


@pytest.mark.parametrize("with_pressure", [False, True])
def test_piso_pressure_and_rhie_chow_policies_preserve_pressure_routes(
    with_pressure,
):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        PDEBoundaryConditions,
        CollocatedPressureSystemControls,
        PisoSolverControls,
        PressureClosureKind,
        resolve_piso_boundary_conditions,
    )
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    right = lambda points: bm.abs(points[..., 0] - 1.0) < 1.0e-12
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: bm.zeros_like(points),
        dirichlet_pressure=(
            (lambda points: bm.ones(points.shape[:-1], dtype=points.dtype))
            if with_pressure
            else None
        ),
        dirichlet_pressure_selector=right if with_pressure else None,
    )
    controls = PisoSolverControls(
        diffusion_method="bounded_over_relaxed",
        diffusion_nonorthogonal_eps=0.1,
    )
    resolved = resolve_piso_boundary_conditions(
        mesh,
        boundary,
        controls,
        CollocatedPressureSystemControls(),
    )
    policy = resolved.pressure_corrector

    assert set(vars(resolved.physical.pressure_state)) == {
        "gradient",
    }
    assert set(vars(policy)) == {
        "dirichlet_operator",
        "gradient",
        "closure",
    }
    expected_closure = (
        PressureClosureKind.DIRICHLET
        if with_pressure
        else PressureClosureKind.NULLSPACE
    )
    assert policy.closure is expected_closure
    np.testing.assert_array_equal(
        policy.dirichlet_operator.faces,
        resolved.physical.pressure_state.gradient.boundary.dirichlet_faces,
    )
    np.testing.assert_array_equal(
        policy.dirichlet_operator.values,
        resolved.physical.pressure_state.gradient.boundary.dirichlet_values,
    )
    assert resolved.rhie_chow.pressure_state is resolved.physical.pressure_state
    assert resolved.rhie_chow.zero_gradient_difference_faces.shape == (0,)
    assert not hasattr(
        resolved.physical.pressure_state,
        "apply_diffusion",
    )
    assert not hasattr(policy, "apply_diffusion")
    assert not hasattr(resolved, "pressure_correction")


def test_high_level_models_enter_solvers_with_resolved_boundaries():
    from fealpy.fvm import (
        NSFVMPISOModel,
        NSFVMSimpleModel,
    )
    from fealpy.fvm.collocated_boundary_conditions import (
        ResolvedPressureSystemBoundary,
        ResolvedVelocityBoundary,
    )

    simple = NSFVMSimpleModel({
        "pde": 2,
        "nx": 2,
        "ny": 2,
        "log_level": "ERROR",
        "pbar_log": False,
    })
    piso = NSFVMPISOModel({
        "pde": 3,
        "nx": 2,
        "ny": 2,
        "time_steps": 1,
        "duration": (0.0, 0.01),
        "log_level": "ERROR",
        "pbar_log": False,
    })

    assert isinstance(
        simple.solver.spatial_face_velocity.boundary,
        ResolvedVelocityBoundary,
    )
    assert isinstance(
        simple.solver.pressure_system.boundary,
        ResolvedPressureSystemBoundary,
    )
    assert isinstance(
        piso.solver.pressure_system.boundary,
        ResolvedPressureSystemBoundary,
    )
    assert not hasattr(simple, "boundary_conditions")
    assert not hasattr(piso, "boundary_conditions")


def test_resolved_physical_boundary_contains_only_solver_inputs():
    from dataclasses import fields

    from fealpy.fvm.collocated_boundary_conditions import (
        ResolvedPhysicalBoundaryConditions,
    )

    assert {field.name for field in fields(ResolvedPhysicalBoundaryConditions)} == {
        "geometry",
        "velocity",
        "momentum",
        "pressure_state",
    }


def test_low_level_solvers_reject_unresolved_pde_boundary():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import (
        CollocatedPisoSolver,
        CollocatedSimpleSolver,
        PDEBoundaryConditions,
        PisoSolverControls,
        SimpleDiscretizationControls,
        SimpleIterationControls,
        build_collocated_ns_linear_solvers,
    )
    from fealpy.mesh import QuadrangleMesh

    bm.set_backend("numpy")
    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: bm.zeros_like(points),
    )
    common = {
        "diffusion_coef": 1.0,
        "convection_coef": 1.0,
        "source": lambda points: bm.zeros_like(points),
        "boundary_conditions": boundary,
    }

    with pytest.raises(
        TypeError,
        match="CollocatedSimpleSolver requires ResolvedSimpleBoundaryConditions",
    ):
        CollocatedSimpleSolver(
            **common,
            discretization_controls=SimpleDiscretizationControls(),
            iteration_controls=SimpleIterationControls(),
            linear_solvers=build_collocated_ns_linear_solvers(),
        )
    with pytest.raises(
        TypeError,
        match="CollocatedPisoSolver requires ResolvedPisoBoundaryConditions",
    ):
        CollocatedPisoSolver(
            **common,
            controls=PisoSolverControls(),
            linear_solvers=build_collocated_ns_linear_solvers(),
        )


def test_piso_model_rejects_traction_before_entering_low_level_solver(
    monkeypatch,
):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel, PDEBoundaryConditions
    import fealpy.fvm.ns_fvm_piso_model as model_module

    def boundary_factory(mesh, pde):
        return PDEBoundaryConditions(
            mesh,
            dirichlet_velocity=pde.dirichlet_velocity,
            momentum_traction=lambda points: bm.zeros_like(points),
            momentum_traction_selector=(
                lambda points: bm.abs(points[..., 0] - 1.0) < 1.0e-12
            ),
        )

    def forbidden_solver_entry(*args, **kwargs):
        raise AssertionError("PISO solver must not receive traction data")

    monkeypatch.setattr(
        model_module.CollocatedPisoSolver,
        "__init__",
        forbidden_solver_entry,
    )

    with pytest.raises(
        ValueError,
        match="momentum traction is not supported.*PISO",
    ):
        NSFVMPISOModel({
            "pde": 3,
            "nx": 2,
            "ny": 2,
            "time_steps": 1,
            "duration": (0.0, 0.01),
            "boundary_conditions": boundary_factory,
            "log_level": "ERROR",
            "pbar_log": False,
        })

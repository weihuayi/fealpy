import numpy as np
import pytest

from fealpy.backend import backend_manager as bm


def _constant_velocity(points):
    return bm.stack(
        [
            bm.ones(points.shape[:-1]),
            bm.zeros(points.shape[:-1]),
        ],
        axis=-1,
    )


def _left_right_engineering_bc(mesh, *, with_pressure=False):
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )

    conditions = [
        BoundaryCondition("velocity", "left", "dirichlet", _constant_velocity),
        BoundaryCondition("velocity", "right", "natural", None),
    ]
    if with_pressure:
        conditions.append(BoundaryCondition("pressure", "right", "dirichlet", 0.0))

    return EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch("left", lambda p: bm.abs(p[..., 0]) < 1.0e-12),
            BoundaryPatch("right", lambda p: bm.abs(p[..., 0] - 1.0) < 1.0e-12),
        ],
        conditions=conditions,
    )


def _left_dirichlet_right_neumann_bc(mesh):
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )

    return EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch("left", lambda p: bm.abs(p[..., 0]) < 1.0e-12),
            BoundaryPatch("right", lambda p: bm.abs(p[..., 0] - 1.0) < 1.0e-12),
        ],
        conditions=[
            BoundaryCondition("velocity", "left", "dirichlet", _constant_velocity),
            BoundaryCondition(
                "velocity",
                "right",
                "neumann",
                lambda p: bm.stack(
                    [
                        2.0 * bm.ones(p.shape[:-1], dtype=p.dtype),
                        3.0 * bm.ones(p.shape[:-1], dtype=p.dtype),
                    ],
                    axis=-1,
                ),
            ),
            BoundaryCondition("pressure", "right", "reference", 5.0),
        ],
    )


def _simple_profile(**discretization_overrides):
    from dataclasses import replace

    from fealpy.fvm import steady_ns_high_accuracy_simple_profile

    base = steady_ns_high_accuracy_simple_profile()
    return replace(
        base,
        discretization=replace(
            base.discretization,
            face_flux_correction_scheme="none",
            face_flux_quadrature_order=3,
            **discretization_overrides,
        ),
    )


def _simple_model_with_left_right_bc(*, with_pressure=False, **options):
    from fealpy.fvm import NSFVMSimpleModel, interpolate_cell_to_face

    def build_bc(mesh, pde):
        return _left_right_engineering_bc(mesh, with_pressure=with_pressure)

    model_options = {
        "pde": 6,
        "nx": 4,
        "ny": 4,
        "profile": _simple_profile(**options),
        "log_level": "ERROR",
        "pbar_log": False,
        "boundary_conditions": build_bc,
    }
    return NSFVMSimpleModel(model_options)


def _simple_model_with_neumann_velocity(**options):
    from fealpy.fvm import NSFVMSimpleModel, interpolate_cell_to_face

    def build_bc(mesh, pde):
        return _left_dirichlet_right_neumann_bc(mesh)

    model_options = {
        "pde": 6,
        "nx": 4,
        "ny": 4,
        "profile": _simple_profile(**options),
        "log_level": "ERROR",
        "pbar_log": False,
        "boundary_conditions": build_bc,
    }
    return NSFVMSimpleModel(model_options)


def _resolved_simple_boundary(mesh, boundary):
    from fealpy.fvm import (
        CollocatedPressureSystemControls,
        SimpleDiscretizationControls,
        resolve_simple_boundary_conditions,
    )

    return resolve_simple_boundary_conditions(
        mesh,
        boundary,
        SimpleDiscretizationControls(),
        CollocatedPressureSystemControls(),
    )


def test_engineering_boundary_conditions_select_only_dirichlet_patches():
    bm.set_backend("numpy")
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)

    bc = _left_right_engineering_bc(mesh)
    pde_bc = bc.to_pde_boundary()

    selected_faces = pde_bc.velocity.dirichlet_faces
    face_centers = pde_bc.geometry.face_center[selected_faces]
    selected_values = pde_bc.velocity.dirichlet_value(
        face_centers
    )

    assert selected_faces.shape[0] == 4
    assert selected_values.shape == (4, 2)
    assert bool(bm.all(mesh.entity_barycenter("face")[selected_faces][:, 0] < 1.0e-12))
    assert bool(bm.all(selected_values[:, 0] == 1.0))
    assert bool(bm.all(selected_values[:, 1] == 0.0))


def test_pde_boundary_values_preserve_arbitrary_point_leading_axes():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=1,
        ny=1,
    )
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: bm.stack(
            (points[..., 0], -points[..., 1]),
            axis=-1,
        ),
        dirichlet_pressure=lambda points: (
            points[..., 0] + 2.0 * points[..., 1]
        ),
    )
    points = bm.array(
        [
            [
                [0.0, 0.25],
                [0.0, 0.75],
                [0.5, 1.0],
            ],
            [
                [1.0, 0.25],
                [1.0, 0.75],
                [0.5, 0.0],
            ],
        ],
        dtype=bm.float64,
    )

    velocity = boundary.velocity.dirichlet_value(points)
    pressure = boundary.pressure.dirichlet_value(points)

    assert velocity.shape == points.shape
    assert pressure.shape == points.shape[:-1]
    np.testing.assert_allclose(velocity[..., 0], points[..., 0])
    np.testing.assert_allclose(velocity[..., 1], -points[..., 1])
    np.testing.assert_allclose(
        pressure,
        points[..., 0] + 2.0 * points[..., 1],
    )


def test_pde_boundary_constant_vector_broadcasts_over_point_leading_axes():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=1,
        ny=1,
    )
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=[2.0, -3.0],
    )
    points = bm.zeros((2, 3, 2), dtype=bm.float64)

    values = boundary.velocity.dirichlet_value(points)

    assert values.shape == points.shape
    np.testing.assert_allclose(values[..., 0], 2.0)
    np.testing.assert_allclose(values[..., 1], -3.0)


def test_engineering_boundary_values_preserve_point_leading_axes():
    bm.set_backend("numpy")
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=1,
        ny=1,
    )
    boundary = EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch(
                "left",
                lambda points: points[..., 0] < 0.5,
            ),
        ],
        conditions=[
            BoundaryCondition(
                "velocity",
                "left",
                "dirichlet",
                lambda points: bm.stack(
                    (1.0 + points[..., 0], points[..., 1]),
                    axis=-1,
                ),
            ),
        ],
    )
    points = bm.array(
        [
            [
                [0.0, 0.25],
                [0.25, 0.75],
            ],
            [
                [0.75, 0.25],
                [1.0, 0.75],
            ],
        ],
        dtype=bm.float64,
    )

    values = boundary.dirichlet_value("velocity")(points)

    assert values.shape == points.shape
    np.testing.assert_allclose(
        values,
        [
            [[1.0, 0.25], [1.25, 0.75]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
    )


def test_vector_boundary_callable_must_return_vector_shape():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=1,
        ny=1,
    )
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: points[..., 0],
    )
    points = boundary.geometry.face_center[
        boundary.velocity.dirichlet_faces
    ]

    with pytest.raises(
        ValueError,
        match=r"velocity boundary callable must return shape \(\.\.\., GD\)",
    ):
        boundary.velocity.dirichlet_value(points)


def test_pde_boundary_resolves_selector_once_and_exposes_only_face_tensors():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    calls = 0

    def left(points):
        nonlocal calls
        calls += 1
        return bm.abs(points[..., 0]) < 1.0e-12

    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=lambda points: bm.zeros_like(points),
        dirichlet_velocity_selector=left,
    )

    faces = boundary.velocity.dirichlet_faces
    assert calls == 1
    assert bm.is_tensor(faces)
    assert faces.ndim == 1


def test_pde_boundary_rejects_overlapping_velocity_face_kinds():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    left = lambda points: bm.abs(points[..., 0]) < 1.0e-12

    with pytest.raises(
        ValueError,
        match="velocity Dirichlet and Neumann faces must be disjoint",
    ):
        PDEBoundaryConditions(
            mesh,
            dirichlet_velocity=lambda points: bm.zeros_like(points),
            dirichlet_velocity_selector=left,
            neumann_velocity=lambda points: bm.zeros_like(points),
            neumann_velocity_selector=left,
        )


def test_pde_boundary_rejects_overlapping_pressure_and_traction_faces():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box(
        [0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    right = lambda points: (
        bm.abs(points[..., 0] - 1.0) < 1.0e-12
    )

    with pytest.raises(
        ValueError,
        match=(
            "pressure Dirichlet/reference and momentum traction faces "
            "must be disjoint"
        ),
    ):
        PDEBoundaryConditions(
            mesh,
            dirichlet_pressure=lambda points: bm.zeros(points.shape[:-1]),
            dirichlet_pressure_selector=right,
            momentum_traction=lambda points: bm.zeros_like(points),
            momentum_traction_selector=right,
        )


def test_engineering_boundary_conditions_convert_to_pde_boundary_conditions():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    engineering_bc = _left_right_engineering_bc(mesh, with_pressure=True)
    pde_bc = engineering_bc.to_pde_boundary()

    boundary_faces = pde_bc.velocity.dirichlet_faces
    face_centers = pde_bc.geometry.face_center[boundary_faces]
    boundary_value = pde_bc.velocity.dirichlet_value(face_centers)
    pressure_faces = pde_bc.pressure.dirichlet_faces

    assert isinstance(pde_bc, PDEBoundaryConditions)
    assert pde_bc.velocity.dirichlet_faces.shape[0] > 0
    assert pde_bc.velocity.natural_faces.shape[0] > 0
    assert pde_bc.pressure.dirichlet_faces.shape[0] > 0
    assert boundary_faces.shape[0] == 4
    assert boundary_value.shape == (4, 2)
    assert bool(bm.all(face_centers[:, 0] < 1.0e-12))
    assert pressure_faces.shape == (4,)


def test_pressure_reference_maps_to_dirichlet_pressure_boundary():
    bm.set_backend("numpy")
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    engineering_bc = _left_dirichlet_right_neumann_bc(mesh)
    pde_bc = engineering_bc.to_pde_boundary()
    pressure_faces = pde_bc.pressure.dirichlet_faces
    pressure_points = pde_bc.geometry.face_center[pressure_faces]
    pressure_value = pde_bc.pressure.dirichlet_value(pressure_points)

    assert pressure_faces.shape == (4,)
    assert bool(bm.all(pressure_value == 5.0))


def test_dirichlet_pressure_without_selector_means_all_boundary_faces():
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=2, ny=2)
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_pressure=lambda points: bm.zeros(points.shape[:-1]),
    )

    np.testing.assert_array_equal(
        bm.to_numpy(boundary.pressure.dirichlet_faces),
        bm.to_numpy(boundary.geometry.boundary_faces),
    )


def test_engineering_boundary_conditions_validate_reference_and_neumann_semantics():
    bm.set_backend("numpy")
    import pytest
    from fealpy.fvm import BoundaryCondition, BoundaryPatch, EngineeringBoundaryConditions
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    patches = [
        BoundaryPatch(
            "right",
            lambda p: bm.abs(p[..., 0] - 1.0) < 1.0e-12,
        )
    ]

    with pytest.raises(ValueError, match="reference.*pressure"):
        EngineeringBoundaryConditions(
            mesh,
            patches,
            [BoundaryCondition("velocity", "right", "reference", 0.0)],
        )
    with pytest.raises(ValueError, match="pressure/neumann"):
        EngineeringBoundaryConditions(
            mesh,
            patches,
            [BoundaryCondition("pressure", "right", "neumann", 0.0)],
        )
    with pytest.raises(ValueError, match="Neumann"):
        EngineeringBoundaryConditions(
            mesh,
            patches,
            [BoundaryCondition("velocity", "right", "neumann", None)],
        )


def test_neumann_velocity_boundary_enters_pde_boundary_protocol():
    bm.set_backend("numpy")
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    pde_bc = _left_dirichlet_right_neumann_bc(mesh).to_pde_boundary()
    neumann_faces = pde_bc.velocity.neumann_faces
    neumann_points = pde_bc.geometry.face_center[neumann_faces]
    neumann_value = pde_bc.velocity.neumann_value(neumann_points)

    assert neumann_faces.shape == (4,)
    assert neumann_value.shape == (4, 2)
    assert bool(bm.all(neumann_value[:, 0] == 2.0))
    assert bool(bm.all(neumann_value[:, 1] == 3.0))


def test_simple_model_uses_engineering_boundary_conditions_for_face_constraints():
    bm.set_backend("numpy")
    from fealpy.fvm import (
        EngineeringBoundaryConditions,
    )
    from fealpy.fvm.collocated_boundary_conditions import (
        ResolvedMomentumBoundary,
        ResolvedPressureSystemBoundary,
        ResolvedVelocityBoundary,
    )

    model = _simple_model_with_left_right_bc()

    boundary_faces = (
        model.solver.spatial_face_velocity.boundary.dirichlet_operator.faces
    )
    boundary_value = (
        model.solver.spatial_face_velocity.boundary.dirichlet_operator.values
    )
    face_centers = model.mesh.entity_barycenter("face")[boundary_faces]

    assert isinstance(model.engineering_bc, EngineeringBoundaryConditions)
    assert isinstance(
        model.solver.spatial_face_velocity.boundary,
        ResolvedVelocityBoundary,
    )
    assert isinstance(
        model.solver.momentum.spatial_operator.momentum_boundary,
        ResolvedMomentumBoundary,
    )
    assert isinstance(
        model.solver.pressure_system.boundary,
        ResolvedPressureSystemBoundary,
    )
    assert boundary_faces.shape[0] == 4
    assert boundary_value.shape == (4, 2)
    assert bool(bm.all(face_centers[:, 0] < 1.0e-12))
    np.testing.assert_array_equal(
        model.solver.spatial_face_velocity.boundary.gradient
        .boundary.dirichlet_faces,
        boundary_faces,
    )


def test_neumann_velocity_boundary_contributes_momentum_rhs():
    bm.set_backend("numpy")

    model = _simple_model_with_neumann_velocity()
    zero_rhs = bm.zeros(
        model.GD * model.NC,
        dtype=model.solver.discretization.geometry.cell_measure.dtype,
    )
    source_cell = model.solver.momentum.spatial_operator.momentum_boundary.diffusion_source(
        model.solver.momentum.spatial_operator.diffusion_coef,
        zero_rhs,
    )
    right_faces = model.engineering_bc.patch_face_index("right")
    owner = model.fvm_geometry.owner[right_faces]
    expected = bm.zeros_like(source_cell)
    face_contribution = model.fvm_geometry.mag_S_f[right_faces, None] * bm.array([2.0, 3.0])
    expected = bm.index_add(expected, owner, face_contribution, axis=0)

    assert float(bm.max(bm.abs(source_cell - expected))) < 1.0e-12


def test_cylinder_case_engineering_boundary_conditions_excludes_outlet_velocity():
    bm.set_backend("numpy")
    import pytest
    from fealpy.fvm import CylinderFlowCase

    pytest.importorskip("gmsh")
    case = CylinderFlowCase(
        mesh_size=0.12,
        cylinder_mesh_size=0.03,
        wake_mesh_size=0.06,
    )
    mesh = case.init_mesh["improved_tri"]()
    bc = case.engineering_boundary_conditions(mesh)
    pde_bc = bc.to_pde_boundary()

    faces = pde_bc.velocity.dirichlet_faces
    face_centers = pde_bc.geometry.face_center[faces]

    assert bool(bm.any(case.is_inlet_boundary(face_centers)))
    assert bool(bm.any(case.is_wall_boundary(face_centers)))
    assert bool(bm.any(case.is_cylinder_boundary(face_centers)))
    assert not bool(bm.any(case.is_outlet_boundary(face_centers)))


def test_simple_dirichlet_pressure_outlet_contributes_pressure_correction_flux():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=True)
    p_corr = bm.ones(model.NC)
    response_coef = bm.ones(model.mesh.number_of_faces())
    flux = model.solver.pressure_system.correction_flux(
        p_corr,
        response_coef,
        pressure_gradient=(
            model.solver.pressure_gradient.cell_gradient(p_corr)
        ),
    )
    outlet_faces = model.engineering_bc.patch_face_index("right")

    assert float(bm.max(bm.abs(flux[outlet_faces]))) > 1.0e-12


def test_simple_pressure_correction_boundary_inherits_diffusion_configuration():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(
        with_pressure=True,
        diffusion_method="bounded_over_relaxed",
        diffusion_nonorthogonal_eps=0.1,
    )

    policy = model.solver.pressure_system.boundary
    assert (
        policy.dirichlet_operator.diffusion_method
        == "bounded_over_relaxed"
    )
    assert policy.dirichlet_operator.nonorthogonal_eps == 0.1
    np.testing.assert_array_equal(
        policy.gradient.boundary.dirichlet_faces,
        policy.dirichlet_operator.faces,
    )


def test_simple_pressure_correction_flux_matches_face_velocity_flux_change():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=False)
    cell_center = model.fvm_geometry.cell_center
    p_corr = cell_center[:, 0] - 0.4 * cell_center[:, 1]
    p_corr_gradient = model.solver.pressure_gradient.cell_gradient(p_corr)
    response_coef = bm.ones(model.mesh.number_of_faces())
    boundary_faces = (
        model.solver.spatial_face_velocity.boundary.dirichlet_operator.faces
    )
    boundary_velocity = (
        model.solver.spatial_face_velocity.boundary.dirichlet_operator.values
    )
    face_velocity = bm.zeros((model.mesh.number_of_faces(), model.GD))
    face_velocity = bm.set_at(face_velocity, boundary_faces, boundary_velocity)
    before_flux = model.solver.spatial_face_velocity.compute_flux(
        face_velocity
    )

    expected_delta = model.solver.pressure_system.correction_flux(
        p_corr,
        response_coef,
        pressure_gradient=p_corr_gradient,
    )
    area = model.fvm_geometry.S_f
    area_squared = bm.einsum("ij,ij->i", area, area)
    corrected = (
        face_velocity
        + (expected_delta / area_squared)[:, None] * area
    )
    corrected = bm.set_at(
        corrected,
        boundary_faces,
        boundary_velocity,
    )
    actual_delta = (
        model.solver.spatial_face_velocity.compute_flux(corrected)
        - before_flux
    )
    internal_faces = bm.nonzero(model.fvm_geometry.is_internal)[0]

    assert float(
        bm.max(bm.abs(actual_delta[internal_faces] - expected_delta[internal_faces]))
    ) < 1.0e-12
    assert float(bm.max(bm.abs(corrected[boundary_faces] - boundary_velocity))) < 1.0e-12


def test_simple_closed_face_flux_has_compatible_pressure_rhs():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=False)
    face_flux = bm.zeros(model.mesh.number_of_faces())
    internal_faces = bm.nonzero(model.fvm_geometry.is_internal)[0]
    face_flux = bm.set_at(
        face_flux,
        internal_faces,
        bm.arange(internal_faces.shape[0], dtype=face_flux.dtype) + 1.0,
    )
    rhs = -model.solver.pressure_equation.divergence_from_flux(face_flux)

    assert abs(float(bm.to_numpy(bm.sum(rhs)))) < 1.0e-12

    incompatible = rhs + 1.0
    projected = model.solver.pressure_equation.project_rhs_to_range(
        incompatible
    )
    assert abs(float(bm.to_numpy(bm.sum(projected)))) < 1.0e-12


def test_simple_pressure_gradient_uses_engineering_dirichlet_pressure_boundary():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=True)
    right_faces = model.engineering_bc.patch_face_index("right")

    np.testing.assert_array_equal(
        model.solver.pressure_gradient.boundary.dirichlet_faces,
        right_faces,
    )


def test_simple_gradient_methods_are_configurable_for_pressure_and_velocity():
    bm.set_backend("numpy")
    import pytest
    from fealpy.fvm import NSFVMSimpleModel

    with pytest.raises(ValueError, match="Unknown cell_gradient variant"):
        NSFVMSimpleModel(
            {
                "pde": 6,
                "nx": 4,
                "ny": 4,
                "profile": _simple_profile(
                    pressure_gradient_method="not_a_gradient_method"
                ),
                "log_level": "ERROR",
                "pbar_log": False,
            }
        )

    with pytest.raises(ValueError, match="Unknown cell_gradient variant"):
        NSFVMSimpleModel(
            {
                "pde": 6,
                "nx": 4,
                "ny": 4,
                "profile": _simple_profile(
                    velocity_gradient_method="not_a_gradient_method"
                ),
                "log_level": "ERROR",
                "pbar_log": False,
            }
        )


def test_simple_face_interpolation_method_controls_response_and_rhie_chow():
    bm.set_backend("numpy")
    import numpy as np
    from fealpy.fvm import NSFVMSimpleModel, interpolate_cell_to_face

    model = NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 4,
            "ny": 4,
            "profile": _simple_profile(
                momentum_face_interpolation="linear",
                pressure_response_interpolation="linear",
                rhie_chow_velocity_interpolation="linear",
            ),
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    ap = bm.ones(model.NC)

    response = np.asarray(
        model.solver.pressure_equation.face_response_coefficient(ap)
    )
    expected = np.asarray(
        interpolate_cell_to_face(
            model.solver.discretization.geometry.cell_measure / ap,
            geometry=model.fvm_geometry,
            method="linear",
        )
    )

    assert np.linalg.norm(response - expected) < 1.0e-12
    assert model.solver.spatial_face_velocity.interpolation == "linear"


def test_rhie_chow_uses_engineering_dirichlet_pressure_boundary():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=True)
    rhie_chow = model.solver.rhie_chow
    pressure = bm.ones(model.NC)
    outlet_faces = model.engineering_bc.patch_face_index("right")
    pressure_gradient = model.solver.pressure_gradient.cell_gradient(
        pressure
    )
    normal_component = bm.einsum(
        "ij,ij->i",
        rhie_chow.pressure_gradient_difference(
            pressure,
            pressure_gradient,
        )[outlet_faces],
        model.fvm_geometry.S_f[outlet_faces],
    )

    assert float(bm.max(bm.abs(normal_component))) > 1.0e-12


def test_simple_natural_velocity_outlet_returns_owner_convection_diagonal():
    bm.set_backend("numpy")

    model = _simple_model_with_left_right_bc(with_pressure=True)
    uf = bm.zeros((model.mesh.number_of_faces(), 2))
    uf = bm.set_at(uf, (slice(None), 0), 1.0)

    diagonal = (
        model.solver.momentum.spatial_operator.momentum_boundary
        .natural_convection_diagonal(
        model.solver.momentum.spatial_operator.convection_coef * uf,
        )
    )
    outlet_faces = model.engineering_bc.patch_face_index("right")
    outlet_owners = model.fvm_geometry.owner[outlet_faces]

    assert diagonal.shape == (model.NC,)
    assert float(bm.min(diagonal[outlet_owners])) > 0.0


def _left_right_traction_engineering_bc(mesh, traction=(0.0, 0.0)):
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )

    return EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch("left", lambda p: bm.abs(p[..., 0]) < 1.0e-12),
            BoundaryPatch("right", lambda p: bm.abs(p[..., 0] - 1.0) < 1.0e-12),
        ],
        conditions=[
            BoundaryCondition("velocity", "left", "dirichlet", _constant_velocity),
            BoundaryCondition("velocity", "right", "natural", None),
            BoundaryCondition("momentum", "right", "traction", traction),
        ],
    )


def test_engineering_boundary_conditions_expose_momentum_traction_face_average():
    bm.set_backend("numpy")
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    engineering = _left_right_traction_engineering_bc(
        mesh,
        traction=(2.0, -3.0),
    )
    boundary = engineering.to_pde_boundary()
    traction = _resolved_simple_boundary(
        mesh,
        boundary,
    ).physical.momentum.traction
    faces = traction.faces
    values = traction.face_average_values
    centers = mesh.entity_barycenter("face")[faces]

    assert boundary.momentum.traction_faces.shape[0] > 0
    assert faces.shape == (2,)
    assert values.shape == (2, 2)
    np.testing.assert_allclose(bm.to_numpy(centers[:, 0]), 1.0)
    np.testing.assert_allclose(
        bm.to_numpy(values),
        [[2.0, -3.0], [2.0, -3.0]],
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        (2.0, [[2.0, 2.0], [2.0, 2.0]]),
        ([2.0, -3.0], [[2.0, -3.0], [2.0, -3.0]]),
        (
            lambda p: bm.stack([p[..., 1], -p[..., 1]], axis=-1),
            [[0.25, -0.25], [0.75, -0.75]],
        ),
    ],
)
def test_momentum_traction_data_follow_vector_broadcast_rules(value, expected):
    bm.set_backend("numpy")
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    boundary = _left_right_traction_engineering_bc(
        mesh,
        traction=value,
    ).to_pde_boundary()
    faces = boundary.momentum.traction_faces
    values = boundary.momentum.traction_value(
        boundary.geometry.face_center[faces]
    )

    np.testing.assert_allclose(bm.to_numpy(values), expected)


@pytest.mark.parametrize("variable", ["velocity", "pressure"])
def test_engineering_boundary_conditions_reject_non_momentum_traction(variable):
    bm.set_backend("numpy")
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=1, ny=1)
    patches = [
        BoundaryPatch(
            "right",
            lambda p: bm.abs(p[..., 0] - 1.0) < 1.0e-12,
        )
    ]

    with pytest.raises(ValueError, match="traction.*momentum"):
        EngineeringBoundaryConditions(
            mesh,
            patches=patches,
            conditions=[
                BoundaryCondition(variable, "right", "traction", [0.0, 0.0])
            ],
        )


@pytest.mark.parametrize("pressure_kind", ["dirichlet", "reference"])
def test_momentum_traction_conflicts_with_pressure_value_on_same_patch(
    pressure_kind,
):
    bm.set_backend("numpy")
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=1, ny=1)

    with pytest.raises(ValueError, match="momentum traction.*pressure"):
        EngineeringBoundaryConditions(
            mesh,
            patches=[
                BoundaryPatch(
                    "right",
                    lambda p: bm.abs(p[..., 0] - 1.0) < 1.0e-12,
                )
            ],
            conditions=[
                BoundaryCondition(
                    "momentum",
                    "right",
                    "traction",
                    [0.0, 0.0],
                ),
                BoundaryCondition("pressure", "right", pressure_kind, 0.0),
            ],
        )


def test_unconfigured_momentum_traction_returns_explicit_empty_state():
    bm.set_backend("numpy")
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    boundary = _left_right_engineering_bc(mesh).to_pde_boundary()
    traction = _resolved_simple_boundary(
        mesh,
        boundary,
    ).physical.momentum.traction

    assert boundary.momentum.traction_faces.shape == (0,)
    assert traction.faces.shape == (0,)
    assert traction.face_average_values.shape == (0, 2)

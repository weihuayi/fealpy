import numpy as np

from fealpy.fem import BilinearForm, LinearForm
from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.mesh import TriangleMesh
from fealpy.fvm.convection_integrator import ConvectionMatrixAssembler
from fealpy.fvm import (
    CollocatedPressureSystemControls,
    ConvectionIntegrator,
    face_interpolation_owner_weight,
    GradientReconstruct,
    NSFVMPISOModel,
    NSFVMSimpleModel,
    ScalarCrossDiffusionIntegrator,
    reconstruct_face_gradient,
    FVMGeometry,
    FaceFluxReconstruct,
    PDEBoundaryConditions,
    ResolvedGradientBoundary,
    SimpleDiscretizationControls,
    resolve_simple_boundary_conditions,
)
from fealpy.fvm.collocated_discretization import CollocatedDiscretization
from fealpy.fvm.collocated_face_velocity_reconstruct import (
    RhieChowInterpolation,
)
from fealpy.fvm.collocated_spatial_face_velocity import (
    CollocatedSpatialFaceVelocity,
)


LINEAR_GRAD = np.array([1.25, -0.5])


def _flow_solver(model):
    return model.solver


def linear_pressure(points):
    return (
        LINEAR_GRAD[0] * points[..., 0]
        + LINEAR_GRAD[1] * points[..., 1]
        + 0.75
    )


def linear_velocity(points):
    u = points[..., 0] + 2.0 * points[..., 1] + 3.0
    v = -0.5 * points[..., 0] + 0.25 * points[..., 1] - 1.0
    return np.stack([u, v], axis=-1)


def _explicit_gradient(geometry, value=None, *, method="layered_lsq"):
    boundary_faces = np.flatnonzero(np.asarray(geometry.is_boundary))
    if value is None:
        faces = boundary_faces[:0]
        values = np.zeros(0, dtype=np.asarray(geometry.cell_center).dtype)
    else:
        faces = boundary_faces
        values = value(np.asarray(geometry.face_center)[faces])
    empty_faces = faces[:0]
    empty_values = np.zeros(
        (0,) + np.asarray(values).shape[1:],
        dtype=np.asarray(values).dtype,
    )
    boundary = ResolvedGradientBoundary(
        dirichlet_faces=faces,
        dirichlet_values=values,
        neumann_faces=empty_faces,
        neumann_sn_grad=empty_values,
    )
    return GradientReconstruct(
        geometry,
        boundary,
        method=method,
    ), boundary


def _resolved_simple_boundary(geometry):
    controls = SimpleDiscretizationControls()
    return resolve_simple_boundary_conditions(
        geometry.mesh,
        PDEBoundaryConditions(
            geometry.mesh,
            dirichlet_velocity=lambda points: np.zeros_like(points),
            geometry=geometry,
        ),
        controls,
        CollocatedPressureSystemControls(),
    )


def _rhie_chow(geometry):
    boundary = _resolved_simple_boundary(geometry)
    return RhieChowInterpolation(
        geometry,
        boundary.rhie_chow,
    )


def _spatial_face_velocity(geometry, *, interpolation="average"):
    boundary = _resolved_simple_boundary(geometry)
    discretization = CollocatedDiscretization(
        geometry=geometry,
    )
    return CollocatedSpatialFaceVelocity(
        discretization=discretization,
        boundary=boundary.physical.velocity,
        scheme="interpolated",
        interpolation=interpolation,
        face_flux_reconstruct=FaceFluxReconstruct(
            geometry=geometry,
            method="none",
        ),
    )


def _skew_two_cell_mesh():
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.2, 1.0],
            [1.5, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return TriangleMesh(nodes, cells)


def _openfoam_owner_weight(mesh, face):
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell)
    owner, neighbour = face_to_cell[face]
    face_center = np.asarray(geometry.face_center[face])
    cell_center = np.asarray(geometry.cell_center)
    sf = np.asarray(geometry.S_f[face])
    own = abs(float(np.dot(sf, face_center - cell_center[owner])))
    nei = abs(float(np.dot(sf, cell_center[neighbour] - face_center)))
    return nei / (own + nei)


def test_convection_integrator_can_use_openfoam_linear_face_weights():
    mesh = _skew_two_cell_mesh()
    space = ScaledMonomialSpace2d(mesh, 0)
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell)
    internal_face = int(
        np.flatnonzero(face_to_cell[:, 0] != face_to_cell[:, 1])[0]
    )
    face_velocity = np.tile(np.array([[0.7, -0.2]]), (mesh.number_of_faces(), 1))

    local = np.asarray(
        ConvectionIntegrator(
            q=2,
            coef=face_velocity,
            interpolation="linear",
        ).assembly(space)
    )

    weight = _openfoam_owner_weight(mesh, internal_face)
    sf = np.asarray(geometry.S_f[internal_face])
    flux = float(np.dot(sf, face_velocity[internal_face]))
    expected = flux * np.array(
        [
            [weight, 1.0 - weight],
            [-weight, -(1.0 - weight)],
        ]
    )

    assert abs(weight - 0.5) > 1.0e-3
    assert np.linalg.norm(local[internal_face] - expected) < 1.0e-12


def test_convection_integrator_expands_face_stencil_for_tensor_space():
    mesh = _skew_two_cell_mesh()
    scalar_space = ScaledMonomialSpace2d(mesh, 0)
    vector_space = TensorFunctionSpace(scalar_space, shape=(2, -1))
    face_velocity = np.tile(np.array([[0.7, -0.2]]), (mesh.number_of_faces(), 1))

    matrix = BilinearForm(vector_space).add_integrator(
        ConvectionIntegrator(q=2, coef=face_velocity)
    ).assembly()

    assert matrix.shape == (2 * mesh.number_of_cells(),) * 2


def test_convection_integrator_default_matches_central_face_stencil():
    mesh = _skew_two_cell_mesh()
    space = ScaledMonomialSpace2d(mesh, 0)
    geometry = FVMGeometry(mesh)
    face_velocity = np.tile(np.array([[0.7, -0.2]]), (mesh.number_of_faces(), 1))

    for interpolation in ("average", "linear"):
        local = ConvectionIntegrator(
            q=2,
            coef=face_velocity,
            interpolation=interpolation,
        ).assembly(space)

        if interpolation == "average":
            owner_weight = np.where(
                np.asarray(geometry.owner != geometry.neighbour),
                0.5,
                1.0,
            )
        else:
            owner_weight = np.asarray(
                face_interpolation_owner_weight(
                    geometry,
                    method="linear",
                )
            )
        flux = np.einsum("ij,ij->i", np.asarray(geometry.S_f), face_velocity)
        expected = flux[:, None, None] * np.array(
            [
                [[weight, 1.0 - weight], [-weight, weight - 1.0]]
                for weight in owner_weight
            ]
        )

        np.testing.assert_allclose(np.asarray(local), expected, rtol=1.0e-13, atol=1.0e-13)


def test_convection_integrator_default_skips_quadrature_fetch(monkeypatch):
    mesh = _skew_two_cell_mesh()
    space = ScaledMonomialSpace2d(mesh, 0)
    face_velocity = np.tile(np.array([[0.7, -0.2]]), (mesh.number_of_faces(), 1))

    def fail_quadrature(*args, **kwargs):
        raise AssertionError("convection assembly should not fetch quadrature data")

    def fail_basis(*args, **kwargs):
        raise AssertionError("convection assembly should not fetch basis data")

    monkeypatch.setattr(mesh, "quadrature_formula", fail_quadrature)
    monkeypatch.setattr(space, "basis", fail_basis)

    local = ConvectionIntegrator(
        q=2,
        coef=face_velocity,
        interpolation="linear",
    ).assembly(space)

    assert local.shape == (mesh.number_of_faces(), 2, 2)


def test_convection_matrix_assembler_matches_bilinear_form_vector_matrix():
    mesh = _skew_two_cell_mesh()
    scalar_space = ScaledMonomialSpace2d(mesh, 0)
    vector_space = TensorFunctionSpace(scalar_space, shape=(2, -1))
    face_velocity = np.array(
        [
            [0.7, -0.2],
            [0.1, 0.4],
            [-0.3, 0.8],
            [0.5, -0.1],
            [-0.6, -0.2],
        ]
    )

    for interpolation in ("average", "linear"):
        reference = BilinearForm(vector_space).add_integrator(
            ConvectionIntegrator(
                q=2,
                coef=face_velocity,
                interpolation=interpolation,
            )
        ).assembly()
        matrix = ConvectionMatrixAssembler(
            vector_space,
            interpolation=interpolation,
        ).assembly(face_velocity)
        diff = matrix.to_scipy() - reference.to_scipy()

        np.testing.assert_allclose(diff.data, 0.0, atol=1.0e-13)


def test_spatial_face_velocity_can_use_openfoam_linear_interpolation():
    mesh = _skew_two_cell_mesh()
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell)
    internal_face = int(
        np.flatnonzero(face_to_cell[:, 0] != face_to_cell[:, 1])[0]
    )
    weight = _openfoam_owner_weight(mesh, internal_face)
    velocity = np.array([[1.0, -2.0], [4.0, 3.0]])
    uf = _spatial_face_velocity(
        geometry,
        interpolation="linear",
    ).interpolate(velocity)

    owner, neighbour = face_to_cell[internal_face]
    expected = weight * velocity[owner] + (1.0 - weight) * velocity[neighbour]

    assert abs(weight - 0.5) > 1.0e-3
    assert np.linalg.norm(np.asarray(uf[internal_face]) - expected) < 1.0e-12


def test_simple_solver_pressure_gradient_integrator_is_exact_for_linear_pressure():
    cases = [
        (NSFVMSimpleModel, {"pde": 6, "nx": 4, "ny": 4}),
    ]

    for model_cls, options in cases:
        model = model_cls({**options, "pbar_log": False})
        solver = _flow_solver(model)
        points = model.fvm_geometry.cell_center
        pressure = linear_pressure(points)

        grad_p = solver.pressure_gradient.cell_gradient(pressure)
        pressure_integrator = np.concatenate([
            np.asarray(solver.discretization.geometry.cell_measure * grad_p[:, 0]),
            np.asarray(solver.discretization.geometry.cell_measure * grad_p[:, 1]),
        ])
        expected = np.concatenate([
            np.asarray(solver.discretization.geometry.cell_measure) * LINEAR_GRAD[0],
            np.asarray(solver.discretization.geometry.cell_measure) * LINEAR_GRAD[1],
        ])

        assert np.linalg.norm(pressure_integrator - expected) < 1.0e-12


def test_piso_velocity_pressure_correction_uses_lsq_gradient_and_cell_layout():
    from fealpy.fvm.collocated_velocity_pressure_coupling import (
        correct_cell_velocity,
    )

    model = NSFVMPISOModel({
        "pde": 3,
        "nx": 4,
        "ny": 4,
        "time_steps": 1,
        "duration": (0.0, 1.0),
        "pbar_log": False,
    })
    solver = model.solver
    points = model.fvm_geometry.cell_center
    pressure_rate = linear_pressure(points)
    velocity = linear_velocity(points)
    response = 0.25
    a_p = np.asarray(model.fvm_geometry.cell_measure) / response
    pressure_gradient = solver.pressure_gradient.cell_gradient(
        pressure_rate
    )

    corrected = np.asarray(
        correct_cell_velocity(
            velocity,
            pressure_gradient,
            solver.discretization.cell_response(a_p),
        )
    )

    expected = np.asarray(velocity) - response * LINEAR_GRAD
    assert np.linalg.norm(corrected - expected) < 1.0e-12


def test_rhie_chow_gradient_difference_vanishes_for_linear_internal_pressure():
    model = NSFVMSimpleModel({
        "pde": 6,
        "nx": 4,
        "ny": 4,
        "pbar_log": False,
    })
    points = model.fvm_geometry.cell_center
    pressure = linear_pressure(points)
    face_to_cell = model.fvm_geometry.face_to_cell
    is_internal = np.asarray(face_to_cell[:, 0] != face_to_cell[:, 1])

    grad_diff = np.asarray(
        model.solver.rhie_chow.pressure_gradient_difference(
            pressure,
            model.solver.pressure_gradient.cell_gradient(pressure),
        )
    )

    assert np.max(np.abs(grad_diff[is_internal])) < 1.0e-12


def test_rhie_chow_interpolation_accepts_precomputed_pressure_gradient():
    model = NSFVMSimpleModel({
        "pde": 6,
        "nx": 4,
        "ny": 4,
        "pbar_log": False,
    })
    pressure = np.zeros(model.NC)
    base_face_velocity = np.zeros(
        (model.solver.discretization.NF, model.GD)
    )
    face_response = np.ones(model.solver.discretization.NF)
    pressure_gradient = np.zeros((model.NC, model.GD))
    rhie_chow = model.solver.rhie_chow

    face_velocity = rhie_chow.apply(
        base_face_velocity,
        pressure,
        face_response,
        pressure_gradient,
    )

    assert face_velocity.shape == (model.mesh.number_of_faces(), model.GD)


def test_cross_diffusion_gradient_path_is_exact_for_linear_velocity():
    model = NSFVMSimpleModel({
        "pde": 6,
        "nx": 4,
        "ny": 4,
        "pbar_log": False,
    })
    points = model.fvm_geometry.cell_center
    velocity = linear_velocity(points)
    exact_cell_grad = np.array([[1.0, 2.0], [-0.5, 0.25]])

    gradient, boundary = _explicit_gradient(
        model.fvm_geometry,
        linear_velocity,
        method="green_gauss",
    )
    cell_grad = np.asarray(
        gradient.cell_gradient(velocity)
    )
    face_grad = np.asarray(reconstruct_face_gradient(
        model.fvm_geometry,
        cell_grad,
        velocity,
        boundary=boundary,
    ))

    assert np.linalg.norm(cell_grad - exact_cell_grad) < 1.0e-12
    internal = np.asarray(model.fvm_geometry.is_internal)
    assert np.linalg.norm(face_grad[internal] - exact_cell_grad) < 1.0e-12


def _boundary_all_cross_diffusion(model, velocity):
    solver = _flow_solver(model)
    space = solver.discretization.velocity_space
    lform = LinearForm(space)
    grad_u = solver.spatial_face_velocity.boundary.gradient.cell_gradient(
        velocity
    )
    grad_f = reconstruct_face_gradient(
        model.fvm_geometry,
        grad_u,
        velocity,
        boundary=solver.spatial_face_velocity.boundary.gradient.boundary,
    )
    lform.add_integrator(
        ScalarCrossDiffusionIntegrator(
            grad_f=grad_f,
            method="bounded_over_relaxed",
            boundary_policy="all",
        )
    )
    return np.asarray(lform.assembly())


def _cell_velocity_for_model(model, points):
    if hasattr(model.pde, "velocity_0"):
        initial_time = (
            model.solver.controls.duration[0]
            if isinstance(model, NSFVMPISOModel)
            else 0.0
        )
        return model.pde.velocity_0(
            points,
            initial_time,
        )
    return model.pde.velocity(points)


def test_momentum_nonorthogonal_rhs_keeps_boundary_correction():
    cases = [
        (NSFVMSimpleModel, {"pde": 6, "nx": 4, "ny": 4}),
        (
            NSFVMPISOModel,
            {
                "pde": 3,
                "nx": 4,
                "ny": 4,
                "time_steps": 1,
                "duration": (0.0, 1.0),
            },
        ),
    ]

    for model_cls, options in cases:
        model = model_cls({**options, "pbar_log": False})
        solver = _flow_solver(model)
        points = model.fvm_geometry.cell_center
        velocity = _cell_velocity_for_model(model, points)

        actual = np.asarray(
            solver.momentum.spatial_operator.nonorthogonal_rhs(velocity)
        )
        expected = _boundary_all_cross_diffusion(model, velocity)

        assert np.linalg.norm(actual - expected) < 1.0e-12


def test_exact_pressure_lsq_gradient_decreases_on_solver_pde_meshes():
    cases = [
        (NSFVMSimpleModel, {"pde": 6}),
    ]

    for model_cls, base_options in cases:
        errors = []
        for nx in (8, 16):
            model = model_cls({
                **base_options,
                "nx": nx,
                "ny": nx,
                "pbar_log": False,
            })
            solver = _flow_solver(model)
            points = model.fvm_geometry.cell_center
            pressure = model.pde.pressure(points)
            grad_p = solver.pressure_gradient.cell_gradient(pressure)
            exact = model.pde.grad_pressure(points)
            errors.append(
                np.sqrt(
                    np.sum(
                        np.asarray(solver.discretization.geometry.cell_measure)[:, None]
                        * (np.asarray(grad_p) - np.asarray(exact)) ** 2
                    )
                )
            )

        assert errors[1] < 0.8 * errors[0]

import numpy as np
import pytest

from fealpy.model.poisson.exp0002 import Exp0002
from fealpy.mesh import (
    HexahedronMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.topology.builder import TopologyBuilder
from fealpy.mesh.view import Mesh
from fealpy.fvm import (
    FVMGeometry,
    GradientReconstruct as _GradientReconstruct,
    ResolvedGradientBoundary,
)
from fealpy.fvm.gradient_reconstruct import (
    LSQGradientReconstruct,
    least_squares_rhs,
)


def _gradient(
    mesh,
    *,
    geometry=None,
    method="layered_lsq",
    boundary_value=None,
    boundary_type="dirichlet",
    boundary_selector=None,
    layer_weights=(1.0, 0.25),
    boundary_weight=1.0,
):
    """Construct the production gradient operator from explicit test data."""
    geometry = FVMGeometry(mesh) if geometry is None else geometry
    boundary_faces = np.flatnonzero(np.asarray(geometry.is_boundary))
    if boundary_value is None:
        selected_faces = boundary_faces[:0]
        selected_values = np.zeros(0, dtype=np.asarray(geometry.cell_center).dtype)
    else:
        face_centers = np.asarray(geometry.face_center)[boundary_faces]
        if boundary_selector is None:
            selected_faces = boundary_faces
        else:
            selected_faces = boundary_faces[
                np.asarray(boundary_selector(face_centers), dtype=bool)
            ]
        selected_values = np.asarray(
            boundary_value(np.asarray(geometry.face_center)[selected_faces])
        )
    empty_faces = selected_faces[:0]
    empty_values = np.zeros(
        (0,) + selected_values.shape[1:],
        dtype=selected_values.dtype,
    )
    if boundary_type == "neumann":
        boundary = ResolvedGradientBoundary(
            dirichlet_faces=empty_faces,
            dirichlet_values=empty_values,
            neumann_faces=selected_faces,
            neumann_sn_grad=selected_values,
        )
    else:
        boundary = ResolvedGradientBoundary(
            dirichlet_faces=selected_faces,
            dirichlet_values=selected_values,
            neumann_faces=empty_faces,
            neumann_sn_grad=empty_values,
        )
    return _GradientReconstruct(
        geometry,
        boundary,
        method=method,
        layer_weights=layer_weights,
        boundary_weight=boundary_weight,
    )


def _mixed_tri_quad_mesh():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
    quadrilaterals = np.array([[1, 4, 5, 3]], dtype=np.int32)

    block = MeshBlock(positions=points)
    block.add_sector(EntitySector("tri", triangles), root=True)
    block.add_sector(EntitySector("quad", quadrilaterals), root=True)
    TopologyBuilder.construct(block)
    return Mesh(block).fealpy_api()


def test_lsq_recovers_linear_gradient_on_quad_and_tri_meshes():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(_gradient(mesh).cell_gradient(field))

        assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_layered_lsq_recovers_linear_gradient_on_mixed_polygon_mesh():
    mesh = _mixed_tri_quad_mesh()
    points = FVMGeometry(mesh).cell_center
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0

    grad = np.asarray(_gradient(mesh).cell_gradient(field))

    expected = np.broadcast_to(np.array([1.0, 2.0]), grad.shape)
    np.testing.assert_allclose(grad, expected, atol=1.0e-12)


def test_variable_face_neighbor_stencil_uses_supplied_fvm_geometry(monkeypatch):
    mesh = _mixed_tri_quad_mesh()
    geometry = FVMGeometry(mesh)
    reconstruct = _gradient(mesh, geometry=geometry)

    def forbidden_face_to_cell(*args, **kwargs):
        raise AssertionError("gradient stencil must reuse FVMGeometry adjacency")

    monkeypatch.setattr(mesh, "face_to_cell", forbidden_face_to_cell)
    stencil = reconstruct.lsq_reconstruct.padded_cell_neighbors(
        mesh.number_of_cells()
    )

    assert stencil.shape[0] == mesh.number_of_cells()


def test_gradient_reconstruct_reuses_supplied_fvm_geometry():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=2, ny=2)
    geometry = FVMGeometry(mesh)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1]

    reconstruct = _gradient(mesh, geometry=geometry)
    grad = np.asarray(reconstruct.cell_gradient(field))

    assert reconstruct.geometry is geometry
    assert not hasattr(reconstruct, "mesh")
    assert not hasattr(reconstruct, "fvm_geometry")
    assert not hasattr(reconstruct, "GD")
    assert not hasattr(reconstruct, "S_f")
    assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_least_squares_rhs_matches_stencil_accumulation_for_scalar_and_vector_fields():
    N = np.array([
        [0, 1, 2, 1],
        [1, 0, 2, 2],
        [2, 0, 1, 0],
    ])
    weighted_d = np.array([
        [[0.0, 0.0], [1.0, 0.2], [0.4, 0.8], [-0.3, 0.5]],
        [[0.0, 0.0], [-0.6, 0.1], [0.7, -0.2], [0.2, 0.9]],
        [[0.0, 0.0], [-0.4, -0.8], [0.3, -0.5], [0.5, 0.4]],
    ])

    scalar = np.array([1.0, 2.5, -0.5])
    expected_scalar = np.zeros((3, 2))
    for k in range(N.shape[1]):
        expected_scalar += (scalar[N[:, k]] - scalar)[:, None] * weighted_d[:, k, :]

    vector = np.stack([scalar, -2.0 * scalar + 0.5], axis=-1)
    expected_vector = np.zeros((3, 2, 2))
    for k in range(N.shape[1]):
        delta = vector[N[:, k]] - vector
        expected_vector += delta[:, :, None] * weighted_d[:, k, None, :]

    np.testing.assert_allclose(least_squares_rhs(scalar, N, weighted_d), expected_scalar)
    np.testing.assert_allclose(least_squares_rhs(vector, N, weighted_d), expected_vector)


def test_layered_lsq_recovers_3d_linear_gradient_on_tetra_mesh():
    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]

    grad = np.asarray(_gradient(mesh).cell_gradient(field))

    assert grad.shape == (mesh.number_of_cells(), mesh.geo_dimension())
    assert np.linalg.norm(grad - np.array([1.0, 2.0, 3.0])) < 1.0e-10


def test_quadratic_lsq_recovers_quadratic_gradient_from_cell_averages():
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=4, ny=4)

    def value(points):
        x, y = points[..., 0], points[..., 1]
        return x**2 + x * y - 0.5 * y**2 + 2.0 * x - y

    def exact_gradient(points):
        x, y = points[..., 0], points[..., 1]
        return np.stack([2.0 * x + y + 2.0, x - y - 1.0], axis=-1)

    geometry = FVMGeometry(mesh)
    cell_average = geometry.cell_integral(
        lambda points, _: value(points), q=4
    ) / geometry.cell_measure
    gradient = np.asarray(
        _gradient(
            mesh,
            method="quadratic_lsq",
            boundary_value=value,
            boundary_type="dirichlet",
        ).cell_gradient(cell_average)
    )
    expected = exact_gradient(geometry.cell_center)

    np.testing.assert_allclose(gradient, expected, atol=2.0e-10)


def test_quadratic_lsq_rejects_rank_deficient_tetrahedral_stencil():
    mesh = TetrahedronMesh.from_box(
        box=[0, 1, 0, 1, 0, 1],
        nx=2,
        ny=2,
        nz=2,
    )

    def value(points):
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        return (
            x**2 + x * y - 0.5 * y**2 + y * z + 0.25 * z**2
            + 2.0 * x - y + 3.0 * z
        )

    geometry = FVMGeometry(mesh)
    cell_average = geometry.cell_integral(
        lambda points, _: value(points),
        q=4,
    ) / geometry.cell_measure
    reconstruct = _gradient(
        mesh,
        method="quadratic_lsq",
        boundary_value=value,
        boundary_type="dirichlet",
    )

    with pytest.raises(ValueError, match="stencil is rank deficient"):
        reconstruct.cell_gradient(cell_average)


@pytest.mark.parametrize(
    "mesh_type",
    ["quadrangle", "hexahedron"],
)
def test_quadratic_lsq_recovers_vector_quadratics_on_tensor_product_cells(mesh_type):
    if mesh_type == "quadrangle":
        mesh = QuadrangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4)
    else:
        mesh = HexahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)

    def scalar_value(points):
        value = points[..., 0] ** 2 + points[..., 0] * points[..., 1]
        if points.shape[-1] == 3:
            value = value + 0.5 * points[..., 2] ** 2
        return value

    def vector_value(points):
        scalar = scalar_value(points)
        return np.stack([scalar, -2.0 * scalar + 1.0], axis=-1)

    geometry = FVMGeometry(mesh)
    cell_average = geometry.cell_integral(
        lambda points, _: vector_value(points), q=4
    ) / geometry.cell_measure[:, None]
    gradient = np.asarray(
        _gradient(
            mesh,
            method="quadratic_lsq",
            boundary_value=vector_value,
            boundary_type="dirichlet",
        ).cell_gradient(cell_average)
    )
    center = np.asarray(geometry.cell_center)
    first = np.zeros_like(center)
    first[:, 0] = 2.0 * center[:, 0] + center[:, 1]
    first[:, 1] = center[:, 0]
    if center.shape[1] == 3:
        first[:, 2] = center[:, 2]
    expected = np.stack([first, -2.0 * first], axis=1)

    np.testing.assert_allclose(gradient, expected, atol=2.0e-10)


def test_face_weighted_lsq_recovers_3d_linear_gradient_on_tetra_interior_cells():
    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]

    def gd(points):
        return points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]

    grad = np.asarray(
        _gradient(mesh, method="face_weighted_lsq", boundary_value=gd).cell_gradient(field)
    )
    geometry = FVMGeometry(mesh)
    interior = np.ones(mesh.number_of_cells(), dtype=bool)
    interior[np.asarray(geometry.owner[geometry.is_boundary])] = False

    assert grad.shape == (mesh.number_of_cells(), mesh.geo_dimension())
    assert np.linalg.norm(grad[interior] - np.array([1.0, 2.0, 3.0])) < 1.0e-10


def test_green_gauss_allocates_3d_gradient_shape_on_tetra_mesh():
    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    field = np.ones(mesh.number_of_cells())

    grad = np.asarray(_gradient(mesh, method="green_gauss").cell_gradient(field))

    assert grad.shape == (mesh.number_of_cells(), mesh.geo_dimension())


def test_layered_lsq_layer_weights_keep_linear_consistency():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(_gradient(
            mesh, layer_weights=(1.0, 0.25)
        ).cell_gradient(field))

        assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_layered_lsq_layer_weights_change_nonlinear_reconstruction():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 + 0.5 * points[:, 1] ** 2
    equal = np.asarray(
        _gradient(mesh, layer_weights=(1.0, 1.0)).cell_gradient(field)
    )
    first_layer_heavy = np.asarray(
        _gradient(mesh, layer_weights=(1.0, 0.05)).cell_gradient(field)
    )

    assert np.linalg.norm(equal - first_layer_heavy) > 1.0e-4


def test_layered_lsq_reuses_cached_matrix_without_dirichlet_boundary():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0
    reconstruct = _gradient(mesh)

    reconstruct.cell_gradient(field)
    cached_matrix = reconstruct.lsq_reconstruct._layered_lsq_cache[2]
    reconstruct.cell_gradient(field)
    matrix = reconstruct.lsq_reconstruct._layered_lsq_cache[2]

    assert matrix is cached_matrix


def test_face_weighted_lsq_reuses_cached_inverse_between_fields(monkeypatch):
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    first_field = points[:, 0] + 2.0 * points[:, 1]
    second_field = points[:, 0] ** 2 - points[:, 1]
    reconstruct = _gradient(mesh, method="face_weighted_lsq")

    call_count = 0
    original = LSQGradientReconstruct.invert_lsq_matrix

    def counting_invert(self, A, method):
        nonlocal call_count
        call_count += 1
        return original(self, A, method)

    monkeypatch.setattr(LSQGradientReconstruct, "invert_lsq_matrix", counting_invert)

    reconstruct.cell_gradient(first_field)
    reconstruct.cell_gradient(second_field)

    assert call_count == 1


def test_layered_lsq_uses_dirichlet_boundary_data_when_given():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros(mesh.number_of_cells())

    def one(points):
        return np.ones(points.shape[0])

    grad = np.asarray(_gradient(
        mesh,
        boundary_value=one,
        boundary_selector=lambda points: np.abs(points[:, 0]) < 1.0e-12,
    ).cell_gradient(field))

    geometry = FVMGeometry(mesh)
    boundary_faces = np.flatnonzero(np.asarray(geometry.is_boundary))
    face_center = np.asarray(geometry.face_center)[boundary_faces]
    owner = np.asarray(geometry.owner)[boundary_faces]
    left_owner = set(np.asarray(owner[np.abs(face_center[:, 0]) < 1.0e-12]).tolist())
    nonzero_owner = set(np.where(np.linalg.norm(grad, axis=1) > 1.0e-12)[0].tolist())

    assert nonzero_owner == left_owner


def test_layered_lsq_boundary_weight_zero_keeps_default_result():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 + 0.5 * points[:, 1] ** 2

    def shifted_value(points):
        return 10.0 + points[:, 0]

    default = np.asarray(_gradient(mesh).cell_gradient(field))
    with_zero_weight = np.asarray(_gradient(
        mesh,
        boundary_value=shifted_value,
        boundary_weight=0.0,
    ).cell_gradient(field))

    assert np.linalg.norm(with_zero_weight - default) < 1.0e-12


def test_layered_lsq_cell_gradient_does_not_consume_neumann_data():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros(mesh.number_of_cells())

    def normal_derivative(points):
        return np.ones(points.shape[0])

    grad = np.asarray(_gradient(
        mesh,
        boundary_value=normal_derivative,
        boundary_type="neumann",
        boundary_selector=lambda points: np.abs(points[:, 0]) < 1.0e-12,
    ).cell_gradient(field))

    np.testing.assert_allclose(grad, 0.0, atol=1.0e-12)


def test_layered_lsq_vector_cell_gradient_does_not_consume_neumann_data():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros((mesh.number_of_cells(), 2))

    def vector_normal_derivative(points):
        return np.stack([
            np.ones(points.shape[0]),
            2.0 * np.ones(points.shape[0]),
        ], axis=-1)

    grad = np.asarray(_gradient(
        mesh,
        boundary_value=vector_normal_derivative,
        boundary_type="neumann",
        boundary_selector=lambda points: np.abs(points[:, 0]) < 1.0e-12,
    ).cell_gradient(field))

    assert grad.shape == (mesh.number_of_cells(), 2, 2)
    np.testing.assert_allclose(grad, 0.0, atol=1.0e-12)


def test_layered_lsq_dirichlet_boundary_data_supports_vector_fields():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros((mesh.number_of_cells(), 2))

    def vector_value(points):
        return np.stack([
            np.ones(points.shape[0]),
            2.0 * np.ones(points.shape[0]),
        ], axis=-1)

    grad = np.asarray(_gradient(
        mesh,
        boundary_value=vector_value,
        boundary_selector=lambda points: np.abs(points[:, 0]) < 1.0e-12,
    ).cell_gradient(field))

    geometry = FVMGeometry(mesh)
    boundary_faces = np.flatnonzero(np.asarray(geometry.is_boundary))
    face_center = np.asarray(geometry.face_center)[boundary_faces]
    owner = np.asarray(geometry.owner)[boundary_faces]
    left_owner = set(np.asarray(owner[np.abs(face_center[:, 0]) < 1.0e-12]).tolist())
    nonzero_owner = set(np.where(np.linalg.norm(grad, axis=(1, 2)) > 1.0e-12)[0].tolist())

    assert grad.shape == (mesh.number_of_cells(), 2, 2)
    assert nonzero_owner == left_owner


def test_layered_lsq_rejects_invalid_layer_weights():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = mesh.entity_barycenter("cell")[:, 0]
    reconstruct = _gradient(mesh)

    with pytest.raises(ValueError, match="layer_weights"):
        _gradient(mesh, layer_weights=(1.0,)).cell_gradient(field)

    with pytest.raises(ValueError, match="non-negative"):
        _gradient(mesh, layer_weights=(1.0, -1.0)).cell_gradient(field)


def test_face_weighted_lsq_matches_reference():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 - 0.5 * points[:, 1] ** 2 + points[:, 0] * points[:, 1]

    def gd(points):
        return points[:, 0] ** 2 - 0.5 * points[:, 1] ** 2 + points[:, 0] * points[:, 1]

    grad = np.asarray(_gradient(
        mesh,
        method="face_weighted_lsq",
        boundary_value=gd,
    ).cell_gradient(field))
    expected = _face_weighted_lsq_reference(mesh, field, boundary_value=gd)

    assert np.linalg.norm(grad - expected) < 1.0e-12


def test_face_weighted_lsq_recovers_linear_gradient_with_boundary_values():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        def gd(points):
            return points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(_gradient(
            mesh,
            method="face_weighted_lsq",
            boundary_value=gd,
        ).cell_gradient(field))

        expected = _face_weighted_lsq_reference(mesh, field, boundary_value=gd)
        interior = _interior_cell_mask(mesh)

        assert np.linalg.norm(grad - expected) < 1.0e-12
        assert np.linalg.norm(grad[interior] - np.array([1.0, 2.0])) < 1.0e-10
        if meshtype == "uniform_quad":
            assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_face_weighted_lsq_supports_vector_fields():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = np.stack([
        points[:, 0] + 2.0 * points[:, 1],
        -0.5 * points[:, 0] + points[:, 1],
    ], axis=-1)

    def gd(points):
        return np.stack([
            points[:, 0] + 2.0 * points[:, 1],
            -0.5 * points[:, 0] + points[:, 1],
        ], axis=-1)

    grad = np.asarray(_gradient(
        mesh,
        method="face_weighted_lsq",
        boundary_value=gd,
    ).cell_gradient(field))

    expected = _face_weighted_lsq_reference(mesh, field, boundary_value=gd)
    interior = _interior_cell_mask(mesh)

    assert grad.shape == (mesh.number_of_cells(), 2, 2)
    assert np.linalg.norm(grad - expected) < 1.0e-12
    assert np.linalg.norm(grad[interior, 0, :] - np.array([1.0, 2.0])) < 1.0e-10
    assert np.linalg.norm(grad[interior, 1, :] - np.array([-0.5, 1.0])) < 1.0e-10


def test_face_weighted_lsq_uses_patch_normal_delta_on_skewed_boundary():
    node = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.4, 1.0],
        [0.0, 1.0],
    ])
    cell = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadrangleMesh(node, cell)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 + 0.5 * points[:, 1]

    def gd(points):
        return points[:, 0] ** 2 + 0.5 * points[:, 1]

    grad = np.asarray(_gradient(
        mesh,
        method="face_weighted_lsq",
        boundary_value=gd,
    ).cell_gradient(field))
    expected = _face_weighted_lsq_reference(
        mesh,
        field,
        boundary_value=gd,
        boundary_delta="normal",
    )
    full_delta = _face_weighted_lsq_reference(
        mesh,
        field,
        boundary_value=gd,
        boundary_delta="full",
    )

    assert np.linalg.norm(full_delta - expected) > 1.0e-3
    assert np.linalg.norm(grad - expected) < 1.0e-12


def test_green_gauss_cell_gradient_does_not_consume_neumann_data():
    node = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.4, 1.0],
        [0.0, 1.0],
    ])
    cell = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadrangleMesh(node, cell)
    field = np.zeros(mesh.number_of_cells())

    def normal_derivative(points):
        return np.ones(points.shape[0])

    grad = np.asarray(_gradient(
        mesh,
        method="green_gauss",
        boundary_value=normal_derivative,
        boundary_type="neumann",
        boundary_selector=lambda points: np.abs(points[:, 1]) < 1.0e-12,
    ).cell_gradient(field))

    np.testing.assert_allclose(grad, 0.0, atol=1.0e-12)


def _face_weighted_lsq_reference(
    mesh,
    U,
    boundary_value=None,
    boundary_type="dirichlet",
    boundary_delta="normal",
):
    U = np.asarray(U)
    geometry = FVMGeometry(mesh)
    NC = geometry.NC
    cell_centers = np.asarray(geometry.cell_center)
    face_centers = np.asarray(geometry.face_center)
    face_to_cell = np.asarray(geometry.face_to_cell)
    owner = face_to_cell[:, 0]
    neighbour = face_to_cell[:, 1]
    is_internal = owner != neighbour
    Sf = np.asarray(geometry.S_f)
    magSf = np.linalg.norm(Sf, axis=1)
    owner_weight = _linear_owner_weight_reference(mesh)

    A = np.zeros((NC, 2, 2), dtype=float)
    if U.ndim == 1:
        b = np.zeros((NC, 2), dtype=float)
    else:
        b = np.zeros((NC, U.shape[1], 2), dtype=float)

    face = np.nonzero(is_internal)[0]
    own = owner[face]
    nei = neighbour[face]
    d = cell_centers[nei] - cell_centers[own]
    scale = magSf[face] / np.einsum("ij,ij->i", d, d)
    outer = scale[:, None, None] * np.einsum("ni,nj->nij", d, d)
    w = owner_weight[face]
    np.add.at(A, own, (1.0 - w)[:, None, None] * outer)
    np.add.at(A, nei, w[:, None, None] * outer)
    delta = U[nei] - U[own]
    if U.ndim == 1:
        rhs = scale[:, None] * delta[:, None] * d
    else:
        rhs = scale[:, None, None] * delta[:, :, None] * d[:, None, :]
    np.add.at(b, own, (1.0 - w)[:, None] * rhs if U.ndim == 1 else (1.0 - w)[:, None, None] * rhs)
    np.add.at(b, nei, w[:, None] * rhs if U.ndim == 1 else w[:, None, None] * rhs)

    boundary_faces = np.flatnonzero(np.asarray(geometry.is_boundary))
    boundary_owner = owner[boundary_faces]
    boundary_d = face_centers[boundary_faces] - cell_centers[boundary_owner]
    if boundary_delta == "normal":
        unit_normal = Sf[boundary_faces] / magSf[boundary_faces, None]
        boundary_d = unit_normal * np.einsum(
            "ij,ij->i", unit_normal, boundary_d
        )[:, None]
    elif boundary_delta != "full":
        raise ValueError("boundary_delta must be 'normal' or 'full'.")
    boundary_scale = magSf[boundary_faces] / np.einsum(
        "ij,ij->i", boundary_d, boundary_d
    )
    boundary_outer = boundary_scale[:, None, None] * np.einsum(
        "ni,nj->nij", boundary_d, boundary_d
    )
    np.add.at(A, boundary_owner, boundary_outer)
    if boundary_value is not None:
        boundary_face_value = boundary_value(face_centers[boundary_faces])
        if boundary_type == "neumann":
            unit_normal = Sf[boundary_faces] / magSf[boundary_faces, None]
            normal_distance = np.abs(
                np.einsum("ij,ij->i", boundary_d, unit_normal)
            )
            boundary_face_value = (
                U[boundary_owner] + boundary_face_value * normal_distance
            )
        boundary_delta_u = boundary_face_value - U[boundary_owner]
        if U.ndim == 1:
            boundary_rhs = (
                boundary_scale[:, None] * boundary_delta_u[:, None] * boundary_d
            )
        else:
            boundary_rhs = (
                boundary_scale[:, None, None]
                * boundary_delta_u[:, :, None]
                * boundary_d[:, None, :]
            )
        np.add.at(b, boundary_owner, boundary_rhs)

    if U.ndim == 1:
        return np.stack([np.linalg.solve(A[cell], b[cell]) for cell in range(NC)])
    return np.stack([
        np.stack([
            np.linalg.solve(A[cell], b[cell, component])
            for cell in range(NC)
        ])
        for component in range(U.shape[1])
    ], axis=1)


def _linear_owner_weight_reference(mesh):
    geometry = FVMGeometry(mesh)
    face_to_cell = np.asarray(geometry.face_to_cell)
    owner = face_to_cell[:, 0]
    neighbour = face_to_cell[:, 1]
    face_centers = np.asarray(geometry.face_center)
    cell_centers = np.asarray(geometry.cell_center)
    Sf = np.asarray(geometry.S_f)
    owner_dist = np.abs(np.einsum("ij,ij->i", Sf, face_centers - cell_centers[owner]))
    neighbour_dist = np.abs(
        np.einsum("ij,ij->i", Sf, cell_centers[neighbour] - face_centers)
    )
    total_dist = owner_dist + neighbour_dist
    weight = np.where(total_dist > 0.0, neighbour_dist / total_dist, 0.5)
    return np.where(owner != neighbour, weight, 1.0)


def _interior_cell_mask(mesh):
    geometry = FVMGeometry(mesh)
    mask = np.ones(geometry.NC, dtype=bool)
    face_to_cell = np.asarray(geometry.face_to_cell)
    boundary_faces = np.flatnonzero(np.asarray(geometry.is_boundary))
    mask[face_to_cell[boundary_faces, 0]] = False
    return mask

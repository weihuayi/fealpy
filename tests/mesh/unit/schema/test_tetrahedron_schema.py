from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import TetrahedronSchema
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view import Mesh
from fealpy.mesher.box import Box3d


def _as_list(value):
    return bm.to_numpy(value).tolist()


def _assert_allclose(actual, expected, message):
    assert bm.allclose(actual, expected), (
        f"{message}\n"
        f"actual: {_as_list(actual)}\n"
        f"expected: {_as_list(expected)}"
    )


def _assert_equal(actual, expected, message):
    actual_list = _as_list(actual)
    expected_list = _as_list(expected)
    assert actual_list == expected_list, (
        f"{message}\n"
        f"actual: {actual_list}\n"
        f"expected: {expected_list}"
    )


def _assert_shape(actual, expected_shape, message):
    assert actual.shape == expected_shape, (
        f"{message}\n"
        f"actual shape: {actual.shape}\n"
        f"expected shape: {expected_shape}"
    )


def _build_single_tet_view():
    bm.set_backend("numpy")
    positions = bm.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=bm.float64,
    )
    tet = bm.asarray([[0, 1, 2, 3]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("tet", tet), root=True)
    mesh = Mesh(block)
    return mesh, mesh.Entity("tet")


def _build_two_tet_view():
    bm.set_backend("numpy")
    positions = bm.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 3.0],
            [1.0, 5.0, 3.0],
            [1.0, 2.0, 7.0],
        ],
        dtype=bm.float64,
    )
    tet = bm.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("tet", tet), root=True)
    mesh = Mesh(block)
    return mesh, mesh.Entity("tet")


class TestTetrahedronSchema:
    """
    TetrahedronSchema 单元测试。

    测试通过新 mesh 入口构造含四面体实体的 Mesh，再通过
    Mesh.Entity("tet") 获得 EntityView，验证四面体形状 Schema
    内部应承担的实体级算法。
    """

    def test_schema_dispatch_and_attributes(self):
        mesh, tet_view = _build_single_tet_view()

        assert tet_view.schema is TetrahedronSchema
        assert tet_view.size() == 1
        assert tet_view.top_dimension() == 3
        assert tet_view.geo_dimension() == mesh.geo_dimension() == 3

        tri_faces = TetrahedronSchema.local_entity("tri")
        edge_faces = TetrahedronSchema.local_entity("segment")

        assert len(tri_faces) == 4
        assert len(edge_faces) == 6
        assert len(TetrahedronSchema.SFace["tri"]) == 4

    def test_schema_only_defines_handoff_methods(self):
        allowed_methods = {
            "barycenter",
            "bc_to_point",
            "shape_function",
            "grad_shape_function_barycentric",
            "grad_shape_function_reference",
            "jacobi_matrix",
            "grad_lambda",
            "multi_index",
            "measure",
            "normal",
            "quadrature_formula",
            "tangent",
        }
        defined_methods = {
            name
            for name, value in TetrahedronSchema.__dict__.items()
            if isinstance(value, classmethod)
        }

        assert defined_methods == allowed_methods

    def test_multi_index_via_schema_behind_user_view(self):
        _, tet_view = _build_single_tet_view()

        _assert_equal(
            tet_view.schema.multi_index((0,)),
            bm.asarray([[0, 0, 0, 0]], dtype=bm.int32),
            "Degree 0 tetrahedron multi-index has one all-zero row",
        )

        mi1 = tet_view.schema.multi_index((1,))
        _assert_shape(mi1, (4, 4), "Degree 1 tetrahedron has four multi-indices")
        _assert_allclose(
            bm.sum(mi1, axis=1),
            bm.ones((4,), dtype=mi1.dtype),
            "Every degree 1 tetrahedron multi-index row must sum to 1",
        )

        mi2 = tet_view.schema.multi_index((2,))
        _assert_shape(mi2, (10, 4), "Degree 2 tetrahedron has ten multi-indices")
        _assert_allclose(
            bm.sum(mi2, axis=1),
            bm.full((10,), 2, dtype=mi2.dtype),
            "Every degree 2 tetrahedron multi-index row must sum to 2",
        )

    def test_multi_index_rejects_invalid_order_argument(self):
        _, tet_view = _build_single_tet_view()

        with pytest.raises(TypeError):
            tet_view.schema.multi_index(2)

        with pytest.raises(ValueError):
            tet_view.schema.multi_index((-1,))

    def test_barycenter_and_measure_through_user_view(self):
        _, tet_view = _build_single_tet_view()

        _assert_allclose(
            tet_view.barycenter(),
            bm.asarray([[0.25, 0.25, 0.25]], dtype=bm.float64),
            "Right tetrahedron barycenter should be the average of its four vertices",
        )
        _assert_allclose(
            tet_view.measure(),
            bm.asarray([1.0 / 6.0], dtype=bm.float64),
            "Right tetrahedron volume should be 1/6",
        )

    def test_bc_to_point_via_schema_behind_user_view(self):
        _, tet_view = _build_single_tet_view()
        ctx = tet_view.context()
        bcs = (
            bm.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                ],
                dtype=bm.float64,
            ),
        )

        points = tet_view.schema.bc_to_point(ctx, bcs, None)

        _assert_shape(
            points,
            (1, 2, 3),
            "bc_to_point maps two barycentric samples for each tetrahedron entity",
        )
        _assert_allclose(
            points,
            bm.asarray([[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]], dtype=bm.float64),
            "Tetrahedron barycentric coordinates should map to physical points",
        )

    def test_grad_lambda_through_user_view(self):
        _, tet_view = _build_single_tet_view()

        grad = tet_view.grad_lambda()

        expected = bm.asarray(
            [[
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]],
            dtype=bm.float64,
        )
        _assert_shape(grad, (1, 4, 3), "Tetrahedron has four barycentric gradients")
        _assert_allclose(grad, expected, "Right tetrahedron barycentric gradients are fixed")
        _assert_allclose(
            bm.sum(grad, axis=1),
            bm.zeros((1, 3), dtype=bm.float64),
            "Barycentric gradient sum must vanish",
        )

    def test_grad_shape_function_and_jacobi_matrix_follow_b_u_x_convention(self):
        _, tet_view = _build_two_tet_view()
        ctx = tet_view.context()
        bcs = (
            bm.asarray(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.10, 0.20, 0.30, 0.40],
                ],
                dtype=bm.float64,
            ),
        )

        grad_b = tet_view.schema.grad_shape_function_barycentric(bcs, (1,))
        grad_u = tet_view.schema.grad_shape_function_reference(bcs, (1,))
        jacobi = tet_view.schema.jacobi_matrix(ctx, bcs, None)
        grad_x = tet_view.grad_shape_function(bcs, p=1, variables="x")

        expected_grad_b = bm.broadcast_to(
            bm.eye(4, dtype=bm.float64)[None, :, :],
            (2, 4, 4),
        )
        expected_grad_u = bm.broadcast_to(
            bm.asarray(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=bm.float64,
            )[None, :, :],
            (2, 4, 3),
        )
        expected_jacobi = bm.asarray(
            [
                [
                    [[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
                ],
                [
                    [[2.0, 0.0, 0.0],
                     [0.0, 3.0, 0.0],
                     [0.0, 0.0, 4.0]],
                    [[2.0, 0.0, 0.0],
                     [0.0, 3.0, 0.0],
                     [0.0, 0.0, 4.0]],
                ],
            ],
            dtype=bm.float64,
        )
        expected_grad_x = bm.asarray(
            [
                [
                    [[-1.0, -1.0, -1.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
                    [[-1.0, -1.0, -1.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
                ],
                [
                    [[-0.5, -1.0 / 3.0, -0.25],
                     [0.5, 0.0, 0.0],
                     [0.0, 1.0 / 3.0, 0.0],
                     [0.0, 0.0, 0.25]],
                    [[-0.5, -1.0 / 3.0, -0.25],
                     [0.5, 0.0, 0.0],
                     [0.0, 1.0 / 3.0, 0.0],
                     [0.0, 0.0, 0.25]],
                ],
            ],
            dtype=bm.float64,
        )

        _assert_shape(grad_b, (2, 4, 4), "Barycentric Tet gradients are [NQ, ldof, num_bc]")
        _assert_allclose(grad_b, expected_grad_b, "Linear Tet barycentric gradients are the identity")
        _assert_shape(grad_u, (2, 4, 3), "Reference Tet gradients are [NQ, ldof, ref_dim]")
        _assert_allclose(grad_u, expected_grad_u, "Linear Tet reference gradients follow lambda(u, v, w)")
        _assert_shape(jacobi, (2, 2, 3, 3), "Tet Jacobian matrices are [NC, NQ, GD, ref_dim]")
        _assert_allclose(jacobi, expected_jacobi, "Affine Tet Jacobian should be constant at all samples")
        _assert_shape(grad_x, (2, 2, 4, 3), "Cartesian Tet gradients are [NC, NQ, ldof, GD]")
        _assert_allclose(grad_x, expected_grad_x, "Cartesian Tet gradients should be J^{-T} scaled")

        grad_b_p2 = tet_view.schema.grad_shape_function_barycentric(bcs, (2,))
        grad_u_p2 = tet_view.schema.grad_shape_function_reference(bcs, (2,))
        _assert_shape(grad_b_p2, (2, 10, 4), "Quadratic Tet barycentric gradients have ten shape functions")
        _assert_shape(grad_u_p2, (2, 10, 3), "Quadratic Tet reference gradients have ten shape functions")

    def test_grad_shape_function_user_api_supports_b_u_x_variables(self, capsys):
        _, tet_view = _build_two_tet_view()
        bcs = bm.asarray(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.10, 0.20, 0.30, 0.40],
            ],
            dtype=bm.float64,
        )

        grad_b = tet_view.grad_shape_function(bcs, p=1, variables="b")
        grad_u = tet_view.grad_shape_function(bcs, p=1, variables="u")
        grad_x = tet_view.grad_shape_function(bcs, p=1, variables="x")

        _assert_shape(grad_b, (2, 4, 4), "User API should expose Tet barycentric gradients")
        _assert_shape(grad_u, (2, 4, 3), "User API should expose Tet reference gradients")
        _assert_shape(grad_x, (2, 2, 4, 3), "User API should expose Tet cartesian gradients")
        assert capsys.readouterr().out == ""

    def test_normal_and_tangent_through_user_view(self):
        _, tet_view = _build_single_tet_view()

        normal = tet_view.normal()
        tangent = tet_view.tangent()

        _assert_shape(
            normal,
            (1, 0, 3),
            "For T=G=3 tetrahedra, normal space has zero directions",
        )
        _assert_shape(
            tangent,
            (1, 3, 3),
            "For T=3 tetrahedra, tangent returns three non-unit directions",
        )
        _assert_allclose(
            tangent,
            bm.asarray([[[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]], dtype=bm.float64),
            "Right tetrahedron tangent basis should be x1-x0, x2-x0, x3-x0",
        )

    def test_scalar_index_keeps_single_entity_axis(self):
        _, tet_view = _build_two_tet_view()

        _assert_allclose(
            tet_view.barycenter(index=1),
            bm.asarray([[1.5, 2.75, 4.0]], dtype=bm.float64),
            "Scalar index should preserve one tetrahedron entity axis in barycenter",
        )
        _assert_allclose(
            tet_view.measure(index=1),
            bm.asarray([4.0], dtype=bm.float64),
            "Scalar index should preserve one tetrahedron entity axis in measure",
        )

        grad = tet_view.grad_lambda(index=1)
        _assert_shape(grad, (1, 4, 3), "Scalar index should preserve grad_lambda entity axis")
        _assert_allclose(
            grad,
            bm.asarray(
                [[
                    [-0.5, -1.0 / 3.0, -0.25],
                    [0.5, 0.0, 0.0],
                    [0.0, 1.0 / 3.0, 0.0],
                    [0.0, 0.0, 0.25],
                ]],
                dtype=bm.float64,
            ),
            "Scalar-indexed affine tetrahedron gradients should match inverse Jacobian rows",
        )

        _assert_shape(
            tet_view.normal(index=1),
            (1, 0, 3),
            "Scalar index should preserve normal entity axis",
        )
        _assert_shape(
            tet_view.tangent(index=1),
            (1, 3, 3),
            "Scalar index should preserve tangent entity axis",
        )

    def test_quadrature_formula_via_schema_behind_user_view(self):
        _, tet_view = _build_single_tet_view()

        qf = tet_view.schema.quadrature_formula(1, qtype="legendre")
        bcs, weights = qf.get_quadrature_points_and_weights()

        _assert_shape(bcs, (1, 4), "Tetrahedron order-1 quadrature has one barycentric point")
        _assert_shape(weights, (1,), "Tetrahedron order-1 quadrature has one weight")
        _assert_allclose(
            bcs,
            bm.asarray([[0.25, 0.25, 0.25, 0.25]], dtype=bm.float64),
            "Order-1 tetrahedron quadrature point should be the barycenter",
        )

    def test_box3d_tetrahedralize_uses_tetrahedron_schema_geometry(self):
        mesh = Box3d(nx=1, ny=1, nz=1).tetrahedralize()
        tet_view = mesh.Entity("tet")

        measures = tet_view.measure()
        barycenters = tet_view.barycenter()
        gradients = tet_view.grad_lambda()

        assert tet_view.size() == 6
        assert bool(bm.all(measures > 0.0))
        _assert_allclose(
            bm.sum(measures),
            bm.asarray(1.0, dtype=bm.float64),
            "Six tetrahedra generated from the unit cube should preserve total volume",
        )
        _assert_shape(barycenters, (6, 3), "Box3d tetrahedra barycenters should be 3D points")
        _assert_shape(gradients, (6, 4, 3), "Each generated tetrahedron has four 3D gradients")

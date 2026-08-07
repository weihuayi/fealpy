from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import HexahedronSchema
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view import Mesh


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


def _build_single_hex_view():
    bm.set_backend("numpy")
    positions = bm.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=bm.float64,
    )
    hex_cell = bm.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("hex", hex_cell), root=True)
    mesh = Mesh(block)
    return mesh, mesh.Entity("hex")


def _build_rectangular_hex_view():
    bm.set_backend("numpy")
    positions = bm.asarray(
        [
            [10.0, -1.0, 2.0],
            [12.0, -1.0, 2.0],
            [12.0, 2.0, 2.0],
            [10.0, 2.0, 2.0],
            [10.0, -1.0, 7.0],
            [12.0, -1.0, 7.0],
            [12.0, 2.0, 7.0],
            [10.0, 2.0, 7.0],
        ],
        dtype=bm.float64,
    )
    hex_cell = bm.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("hex", hex_cell), root=True)
    mesh = Mesh(block)
    return mesh, mesh.Entity("hex")


class TestHexahedronSchema:
    def test_schema_dispatch_and_attributes(self):
        mesh, hex_view = _build_single_hex_view()

        assert hex_view.schema is HexahedronSchema
        assert hex_view.size() == 1
        assert hex_view.top_dimension() == 3
        assert hex_view.geo_dimension() == mesh.geo_dimension() == 3

        assert len(HexahedronSchema.OFace["quad"]) == 6
        assert len(HexahedronSchema.SFace["quad"]) == 6

    def test_schema_only_defines_handoff_methods(self):
        allowed_methods = {
            "barycenter",
            "bc_to_point",
            "shape_function",
            "grad_shape_function_barycentric",
            "grad_shape_function_reference",
            "jacobi_matrix",
            "multi_index",
            "grad_lambda",
            "measure",
            "normal",
            "quadrature_formula",
            "tangent",
        }
        defined_methods = {
            name
            for name, value in HexahedronSchema.__dict__.items()
            if isinstance(value, classmethod)
        }

        assert defined_methods == allowed_methods

    def test_multi_index_requires_tuple_order(self):
        _, hex_view = _build_single_hex_view()

        mi = hex_view.schema.multi_index((2,))
        _assert_shape(mi, (27, 8), "Degree 2 hexahedron multi-index has one column per vertex")
        _assert_allclose(
            bm.sum(mi, axis=1),
            bm.full((27,), 8, dtype=mi.dtype),
            "Degree 2 tensor-product Hex vertex multi-index rows sum to 2^3",
        )

        mi_raw = hex_view.schema.multi_index((2,), tensorprod=False)
        _assert_shape(mi_raw, (27, 6), "Raw degree 2 hexahedron multi-index has interval pairs")
        _assert_allclose(
            bm.min(mi_raw, axis=0),
            bm.asarray([0, 0, 0, 0, 0, 0], dtype=mi_raw.dtype),
            "Each degree 2 tensor-product direction starts at 0",
        )
        _assert_allclose(
            bm.max(mi_raw, axis=0),
            bm.asarray([2, 2, 2, 2, 2, 2], dtype=mi_raw.dtype),
            "Each degree 2 tensor-product direction reaches order 2",
        )

        anisotropic = hex_view.schema.multi_index((1, 2, 3))
        _assert_shape(anisotropic, (24, 8), "Anisotropic hexahedron vertex multi-index has product size")

        anisotropic_raw = hex_view.schema.multi_index((1, 2, 3), tensorprod=False)
        _assert_shape(anisotropic_raw, (24, 6), "Anisotropic raw hexahedron multi-index has interval pairs")

        with pytest.raises(TypeError):
            hex_view.schema.multi_index(2)

        with pytest.raises(ValueError):
            hex_view.schema.multi_index((-1,))

    def test_barycenter_measure_and_bc_to_point(self):
        _, hex_view = _build_single_hex_view()
        ctx = hex_view.context()

        _assert_allclose(
            hex_view.barycenter(),
            bm.asarray([[0.5, 0.5, 0.5]], dtype=bm.float64),
            "Unit hexahedron barycenter should be the average of its eight vertices",
        )
        _assert_allclose(
            hex_view.measure(),
            bm.asarray([1.0], dtype=bm.float64),
            "Unit hexahedron measure should be one",
        )

        bcs = (
            bm.asarray([[0.2, 0.8], [0.65, 0.35]], dtype=bm.float64),
            bm.asarray([[0.7, 0.3], [0.1, 0.9]], dtype=bm.float64),
            bm.asarray([[0.4, 0.6], [0.75, 0.25]], dtype=bm.float64),
        )
        points = hex_view.schema.bc_to_point(ctx, bcs, None)

        _assert_shape(
            points,
            (1, 8, 3),
            "Tensor-product barycentric coordinates map to flattened (NC, NQ, GD) points",
        )
        _assert_allclose(
            points[0, 0],
            bm.asarray([0.8, 0.3, 0.6], dtype=bm.float64),
            "Non-symmetric tensor-product coordinates should map as (u1, v1, w1)",
        )
        _assert_allclose(
            points[0, 7],
            bm.asarray([0.35, 0.9, 0.25], dtype=bm.float64),
            "Second tensor-product sample should preserve each coordinate direction",
        )

    def test_rectangular_hex_geometry_uses_physical_scales(self):
        _, hex_view = _build_rectangular_hex_view()
        ctx = hex_view.context()

        _assert_allclose(
            hex_view.barycenter(),
            bm.asarray([[11.0, 0.5, 4.5]], dtype=bm.float64),
            "Rectangular hexahedron barycenter should average the eight physical vertices",
        )
        _assert_allclose(
            hex_view.measure(),
            bm.asarray([30.0], dtype=bm.float64),
            "Rectangular hexahedron volume should be dx * dy * dz",
        )

        bcs = (
            bm.asarray([[0.2, 0.8]], dtype=bm.float64),
            bm.asarray([[0.7, 0.3]], dtype=bm.float64),
            bm.asarray([[0.4, 0.6]], dtype=bm.float64),
        )
        points = hex_view.schema.bc_to_point(ctx, bcs, None)
        _assert_allclose(
            points[0, 0],
            bm.asarray([11.6, -0.1, 5.0], dtype=bm.float64),
            "Tensor-product coordinates should scale and translate in physical space",
        )

    def test_grad_shape_function_and_jacobi_matrix_follow_b_u_x_convention(self):
        _, hex_view = _build_rectangular_hex_view()
        ctx = hex_view.context()
        bcs = (
            bm.asarray([[0.2, 0.8]], dtype=bm.float64),
            bm.asarray([[0.7, 0.3]], dtype=bm.float64),
            bm.asarray([[0.4, 0.6]], dtype=bm.float64),
        )

        grad_b = hex_view.schema.grad_shape_function_barycentric(bcs, (1, 1, 1))
        grad_u = hex_view.schema.grad_shape_function_reference(bcs, (1, 1, 1))
        jacobi = hex_view.schema.jacobi_matrix(ctx, bcs, None)
        grad_x = hex_view.grad_shape_function(bcs, p=1, variables="x")

        expected_grad_b = bm.asarray(
            [[
                [0.28, 0.0, 0.08, 0.0, 0.14, 0.0],
                [0.0, 0.28, 0.32, 0.0, 0.56, 0.0],
                [0.0, 0.12, 0.0, 0.32, 0.24, 0.0],
                [0.12, 0.0, 0.0, 0.08, 0.06, 0.0],
                [0.42, 0.0, 0.12, 0.0, 0.0, 0.14],
                [0.0, 0.42, 0.48, 0.0, 0.0, 0.56],
                [0.0, 0.18, 0.0, 0.48, 0.0, 0.24],
                [0.18, 0.0, 0.0, 0.12, 0.0, 0.06],
            ]],
            dtype=bm.float64,
        )
        expected_grad_u = bm.asarray(
            [[
                [-0.28, -0.08, -0.14],
                [0.28, -0.32, -0.56],
                [0.12, 0.32, -0.24],
                [-0.12, 0.08, -0.06],
                [-0.42, -0.12, 0.14],
                [0.42, -0.48, 0.56],
                [0.18, 0.48, 0.24],
                [-0.18, 0.12, 0.06],
            ]],
            dtype=bm.float64,
        )
        expected_jacobi = bm.asarray(
            [[[
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 5.0],
            ]]],
            dtype=bm.float64,
        )
        expected_grad_x = bm.asarray(
            [[[
                [-0.14, -0.08 / 3.0, -0.028],
                [0.14, -0.32 / 3.0, -0.112],
                [0.06, 0.32 / 3.0, -0.048],
                [-0.06, 0.08 / 3.0, -0.012],
                [-0.21, -0.04, 0.028],
                [0.21, -0.16, 0.112],
                [0.09, 0.16, 0.048],
                [-0.09, 0.04, 0.012],
            ]]],
            dtype=bm.float64,
        )

        _assert_shape(grad_b, (1, 8, 6), "Barycentric Hex gradients are [NQ, ldof, 6]")
        _assert_allclose(grad_b, expected_grad_b, "Linear Hex barycentric gradients follow product rule")
        _assert_shape(grad_u, (1, 8, 3), "Reference Hex gradients are [NQ, ldof, ref_dim]")
        _assert_allclose(grad_u, expected_grad_u, "Linear Hex reference gradients follow interval chain rule")
        _assert_shape(jacobi, (1, 1, 3, 3), "Hex Jacobian matrices are [NC, NQ, GD, ref_dim]")
        _assert_allclose(jacobi, expected_jacobi, "Rectangular Hex Jacobian should contain physical side lengths")
        _assert_shape(grad_x, (1, 1, 8, 3), "Cartesian Hex gradients are [NC, NQ, ldof, GD]")
        _assert_allclose(grad_x, expected_grad_x, "Cartesian Hex gradients should be J^{-T} scaled")

    def test_grad_shape_function_supports_higher_and_anisotropic_orders(self):
        _, hex_view = _build_rectangular_hex_view()
        bcs = (
            bm.asarray([[0.2, 0.8]], dtype=bm.float64),
            bm.asarray([[0.7, 0.3]], dtype=bm.float64),
            bm.asarray([[0.4, 0.6]], dtype=bm.float64),
        )

        grad_b_p2 = hex_view.grad_shape_function(bcs, p=2, variables="b")
        grad_u_p2 = hex_view.grad_shape_function(bcs, p=2, variables="u")
        grad_b_aniso = hex_view.grad_shape_function(bcs, p=(1, 2, 3), variables="b")
        grad_u_aniso = hex_view.grad_shape_function(bcs, p=(1, 2, 3), variables="u")

        _assert_shape(grad_b_p2, (1, 27, 6), "Quadratic Hex barycentric gradients have 27 shape functions")
        _assert_shape(grad_u_p2, (1, 27, 3), "Quadratic Hex reference gradients have 27 shape functions")
        _assert_shape(grad_b_aniso, (1, 24, 6), "Anisotropic Hex barycentric gradients use product size")
        _assert_shape(grad_u_aniso, (1, 24, 3), "Anisotropic Hex reference gradients use product size")

    def test_grad_shape_function_user_api_supports_b_u_x_variables(self):
        _, hex_view = _build_rectangular_hex_view()
        bcs = (
            bm.asarray([[0.2, 0.8]], dtype=bm.float64),
            bm.asarray([[0.7, 0.3]], dtype=bm.float64),
            bm.asarray([[0.4, 0.6]], dtype=bm.float64),
        )

        grad_b = hex_view.grad_shape_function(bcs, p=1, variables="b")
        grad_u = hex_view.grad_shape_function(bcs, p=1, variables="u")
        grad_x = hex_view.grad_shape_function(bcs, p=1, variables="x")

        _assert_shape(grad_b, (1, 8, 6), "User API should expose Hex barycentric gradients")
        _assert_shape(grad_u, (1, 8, 3), "User API should expose Hex reference gradients")
        _assert_shape(grad_x, (1, 1, 8, 3), "User API should expose Hex cartesian gradients")

    def test_normal_tangent_and_quadrature(self):
        _, hex_view = _build_single_hex_view()

        _assert_shape(
            hex_view.normal(),
            (1, 0, 3),
            "For T=G=3 hexahedra, normal space has zero directions",
        )
        _assert_allclose(
            hex_view.tangent(),
            bm.asarray([[[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]], dtype=bm.float64),
            "Unit hexahedron tangent basis should follow local x, y, z edges",
        )

        qf = hex_view.schema.quadrature_formula(2)
        bcs, weights = qf.get_quadrature_points_and_weights()

        assert isinstance(bcs, tuple)
        assert len(bcs) == 3
        _assert_shape(weights, (8,), "Order 2 hexahedron tensor quadrature has eight weights")
        _assert_allclose(
            bm.sum(weights),
            bm.asarray(1.0, dtype=weights.dtype),
            "Reference hexahedron tensor quadrature weights should sum to one",
        )

    def test_rectangular_hex_tangent_uses_non_unit_edges(self):
        _, hex_view = _build_rectangular_hex_view()

        _assert_allclose(
            hex_view.tangent(),
            bm.asarray([[[2.0, 0.0, 0.0],
                         [0.0, 3.0, 0.0],
                         [0.0, 0.0, 5.0]]], dtype=bm.float64),
            "Tangent directions should preserve non-unit physical edge vectors",
        )

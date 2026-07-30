import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import PyramidSchema
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view import Mesh


PYRAMID_DATA = [
    {
        "point": bm.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=bm.float64,
        ),
        "cell": bm.asarray([[0, 1, 2, 3, 4]], dtype=bm.int64),
        "measure": bm.asarray([1.0 / 3.0], dtype=bm.float64),
        "barycenter": bm.asarray([[0.5, 0.5, 0.2]], dtype=bm.float64),
        "normal_shape": (1, 0, 3),
        "tangent_shape": (1, 3, 3),
        "local_edges": [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 4],
            [1, 4],
            [2, 4],
            [3, 4],
        ],
    }
]


def _assert_allclose(actual, expected, message, atol=1.0e-14):
    assert bm.allclose(actual, expected, atol=atol), (
        f"{message}\n"
        f"actual: {actual}\n"
        f"expected: {expected}"
    )


def _build_pyramid_mesh(data) -> Mesh:
    block = MeshBlock(positions=data["point"])
    block.add_sector(EntitySector("pyramid", data["cell"]), root=True)
    return Mesh(block)


def _interval_bc(t: float):
    return bm.asarray([[1.0 - t, t]], dtype=bm.float64)


def _pyramid_bcs(u: float, v: float, w: float):
    return (_interval_bc(u), _interval_bc(v), _interval_bc(w))


class TestPyramidMesh:
    """
    Unit tests for pyramid entity schema rules and first-stage geometric methods.
    """

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_schema_geometry_on_standard_cell(self, data) -> None:
        """
        Verify volume, vertex-average barycenter, normal shape, and tangent shape.
        """
        pyramid = _build_pyramid_mesh(data).Entity("pyramid")

        _assert_allclose(
            pyramid.measure(),
            data["measure"],
            "Standard pyramid volume should be 1/3.",
        )
        _assert_allclose(
            pyramid.barycenter(),
            data["barycenter"],
            "Pyramid barycenter follows the current vertex-average convention.",
        )

        normal = pyramid.normal()
        tangent = pyramid.tangent()
        assert normal.shape == data["normal_shape"]
        assert tangent.shape == data["tangent_shape"]
        assert bm.linalg.matrix_rank(tangent[0]) == 3

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_schema_infers_local_edges(self, data) -> None:
        """
        Verify that pyramid local edges are inferred from quad/tri local faces.
        """
        assert PyramidSchema.local_entity("segment") == data["local_edges"]

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_geometry_shape_function_and_bc_to_point(self, data) -> None:
        """
        Verify collapsed-coordinate geometry shape functions and physical mapping.
        """
        pyramid = _build_pyramid_mesh(data).Entity("pyramid")
        bcs = _pyramid_bcs(0.25, 0.75, 0.5)

        phi = PyramidSchema.geometry_shape_function(bcs)
        _assert_allclose(
            phi,
            bm.asarray([[0.09375, 0.03125, 0.09375, 0.28125, 0.5]], dtype=bm.float64),
            "Collapsed pyramid shape-function values should match hand calculation.",
        )
        _assert_allclose(
            bm.sum(phi, axis=1),
            bm.ones(1, dtype=bm.float64),
            "Pyramid geometry shape functions should form a partition of unity.",
        )

        point = PyramidSchema.bc_to_point(pyramid.context(), bcs, None)
        _assert_allclose(
            point,
            bm.asarray([[[0.375, 0.625, 0.5]]], dtype=bm.float64),
            "Pyramid reference point should map to the expected physical point.",
        )

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_reference_vertices_map_to_physical_vertices(self, data) -> None:
        """
        Verify vertex interpolation and collapsed apex layer.
        """
        pyramid = _build_pyramid_mesh(data).Entity("pyramid")

        reference_vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        for i, ref_vertex in enumerate(reference_vertices):
            point = PyramidSchema.bc_to_point(
                pyramid.context(), _pyramid_bcs(*ref_vertex), None
            )
            _assert_allclose(
                point[0, 0],
                data["point"][i],
                "Pyramid reference vertices should map to matching physical vertices.",
            )

        apex_layer_point = PyramidSchema.bc_to_point(
            pyramid.context(), _pyramid_bcs(0.37, 0.61, 1.0), None
        )
        _assert_allclose(
            apex_layer_point[0, 0],
            data["point"][4],
            "The collapsed w=1 layer should map to the apex.",
        )

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_jacobi_matrix_and_grad_transform(self, data) -> None:
        """
        Verify reference derivatives, Jacobian, and chain-rule gradient transform.
        """
        pyramid = _build_pyramid_mesh(data).Entity("pyramid")
        bcs = _pyramid_bcs(0.5, 0.5, 0.5)

        gphi = PyramidSchema.geometry_grad_shape_function(bcs)
        _assert_allclose(
            bm.sum(gphi, axis=1),
            bm.zeros((1, 3), dtype=bm.float64),
            "Derivatives of partition-of-unity shape functions should sum to zero.",
        )

        J = PyramidSchema.jacobi_matrix(pyramid.context(), bcs, None)
        _assert_allclose(
            J,
            bm.asarray(
                [[[[0.5, 0.0, 0.0],
                   [0.0, 0.5, 0.0],
                   [0.0, 0.0, 1.0]]]],
                dtype=bm.float64,
            ),
            "Standard pyramid center Jacobian should match hand calculation.",
        )

        ref_grad = bm.asarray(
            [[[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]]],
            dtype=bm.float64,
        )
        grad = PyramidSchema.transform_grad(pyramid.context(), bcs, ref_grad, None)
        _assert_allclose(
            grad,
            bm.asarray(
                [[[[2.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0],
                   [0.0, 0.0, 1.0]]]],
                dtype=bm.float64,
            ),
            "Gradient transform should match the inverse-Jacobian calculation.",
        )

    def test_pyramid_geometry_gradient_matches_finite_difference(self) -> None:
        """
        Verify reference derivatives against finite differences of shape functions.
        """
        u, v, w = 0.23, 0.41, 0.35
        eps = 1.0e-6

        def phi_at(a, b, c):
            return PyramidSchema.geometry_shape_function(_pyramid_bcs(a, b, c))[0]

        finite_diff = bm.stack(
            [
                (phi_at(u + eps, v, w) - phi_at(u - eps, v, w)) / (2.0 * eps),
                (phi_at(u, v + eps, w) - phi_at(u, v - eps, w)) / (2.0 * eps),
                (phi_at(u, v, w + eps) - phi_at(u, v, w - eps)) / (2.0 * eps),
            ],
            axis=-1,
        )
        gphi = PyramidSchema.geometry_grad_shape_function(_pyramid_bcs(u, v, w))[0]
        _assert_allclose(
            gphi,
            finite_diff,
            "Analytic geometry gradients should match central finite differences.",
            atol=1.0e-10,
        )

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_transform_grad_matches_inverse_jacobian(self, data) -> None:
        """
        Verify transform_grad against the explicit inverse-Jacobian formula.
        """
        pyramid = _build_pyramid_mesh(data).Entity("pyramid")
        bcs = _pyramid_bcs(0.2, 0.7, 0.4)
        ref_grad = bm.asarray(
            [[[1.0, 2.0, -0.5],
              [-0.25, 0.75, 1.5]]],
            dtype=bm.float64,
        )

        J = PyramidSchema.jacobi_matrix(pyramid.context(), bcs, None)[0, 0]
        grad = PyramidSchema.transform_grad(pyramid.context(), bcs, ref_grad, None)[0, 0]
        expected = bm.einsum("ik,kd->id", ref_grad[0], bm.linalg.inv(J))
        _assert_allclose(
            grad,
            expected,
            "transform_grad should match ref_grad @ inv(J) when GD == TD.",
        )

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_quadrature_integrates_jacobian_determinant(self, data) -> None:
        """
        Verify collapsed tensor-product quadrature recovers the physical volume.
        """
        pyramid = _build_pyramid_mesh(data).Entity("pyramid")
        for q in (1, 2):
            qf = PyramidSchema.quadrature_formula(q)
            bcs, ws = qf.get_quadrature_points_and_weights()

            J = PyramidSchema.jacobi_matrix(pyramid.context(), bcs, None)[0]
            detJ = bm.linalg.det(J)
            volume = bm.sum(ws * detJ)
            _assert_allclose(
                volume,
                data["measure"][0],
                "Quadrature of det(J) should recover standard pyramid volume.",
            )

    def test_pyramid_quadrature_rejects_unsupported_type(self) -> None:
        """
        Verify that only the current legendre rule is accepted.
        """
        with pytest.raises(ValueError):
            PyramidSchema.quadrature_formula(2, qtype="lobatto")

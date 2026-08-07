from importlib.util import find_spec
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import SegmentSchema
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.ipoints import multi_index_sort
from fealpy.mesh.view import Mesh


BACKENDS = ["numpy"]
BACKEND_IDS = ["backend-numpy"]

if find_spec("torch") is not None:
    BACKENDS.append("pytorch")
    BACKEND_IDS.append("backend-pytorch")


def _as_list(value):
    return bm.to_numpy(value).tolist()


def _assert_allclose(actual, expected, message):
    assert bm.allclose(actual, expected), (
        f"{message}\n"
        f"actual: {_as_list(actual)}\n"
        f"expected: {_as_list(expected)}")


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


def _build_segment_view():
    positions = bm.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=bm.float64,
    )
    edge = bm.asarray(
        [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
        ],
        dtype=bm.int64,
    )
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("segment", edge), root=True)
    mesh = Mesh(block)
    return mesh, mesh.Entity("segment")




def _build_3d_segment_view():
    positions = bm.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [2.0, 4.0, 5.0],
        ],
        dtype=bm.float64,
    )
    edge = bm.asarray([[0, 1], [2, 3], [4, 5]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("segment", edge), root=True)
    mesh = Mesh(block)
    return mesh, mesh.Entity("segment")


def _build_degenerate_segment_view():
    positions = bm.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=bm.float64,
    )
    edge = bm.asarray([[0, 1]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("segment", edge), root=True)
    mesh = Mesh(block)
    return mesh, mesh.Entity("segment")

def _build_two_segment_view():
    positions = bm.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [2.0, 5.0],
        ],
        dtype=bm.float64,
    )
    edge = bm.asarray([[0, 1], [2, 3]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("segment", edge), root=True)
    mesh = Mesh(block)
    return mesh, mesh.Entity("segment")


class TestSegmentSchema:
    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_schema_dispatch_and_attributes(self, backend):
        bm.set_backend(backend)
        mesh, segment_view = _build_segment_view()

        assert segment_view.schema is SegmentSchema
        assert segment_view.size() == 4
        assert segment_view.top_dimension() == 1
        assert segment_view.geo_dimension() == mesh.geo_dimension() == 2
        assert SegmentSchema.local_entity("point") == [[0], [1]]
        assert SegmentSchema.OFace == {"point": [[0], [1]]}
        assert SegmentSchema.SFace == {"point": [[0], [1]]}

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_multi_index_via_schema_behind_user_view(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_segment_view()

        _assert_equal(
            segment_view.schema.multi_index((0,)),
            bm.asarray([[0, 0]], dtype=bm.int32),
            "Degree 0 edge multi-index has one all-zero row",
        )
        _assert_equal(
            segment_view.schema.multi_index((2,)),
            bm.asarray([[2, 0], [1, 1], [0, 2]], dtype=bm.int32),
            "Degree 2 edge multi-index should enumerate the two-vertex simplex indices",
        )
        assert segment_view.schema.num_multi_index((2,)) == 3

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_multi_index_rejects_invalid_order_argument(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_segment_view()

        with pytest.raises(TypeError):
            segment_view.schema.multi_index(2)

        with pytest.raises(ValueError):
            segment_view.schema.multi_index((-1,))

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_multi_index_sort_returns_lexicographic_order(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_segment_view()

        multi_index = bm.asarray(
            [
                [0, 2],
                [2, 0],
                [1, 1],
            ],
            dtype=bm.int32,
        )
        _assert_equal(
            multi_index_sort(multi_index),
            bm.asarray([1, 0, 2], dtype=bm.int64),
            "Edge multi-index rows should follow the topology helper's predefined orientation order",
        )

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_barycenter_measure_grad_lambda_through_user_view(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_segment_view()

        _assert_allclose(
            segment_view.barycenter(),
            bm.asarray([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0, 0.5]], dtype=bm.float64),
            "Edge barycenter should be the average of its two endpoints",
        )
        _assert_allclose(
            segment_view.measure(),
            bm.asarray([1.0, 1.0, 2.0**0.5, 1.0], dtype=bm.float64),
            "Edge measure should be endpoint distance",
        )
        _assert_allclose(
            segment_view.grad_lambda(),
            bm.asarray(
                [
                    [[-1.0, -0.0], [1.0, 0.0]],
                    [[-0.0, -1.0], [0.0, 1.0]],
                    [[0.5, -0.5], [-0.5, 0.5]],
                    [[-0.0, -1.0], [0.0, 1.0]],
                ],
                dtype=bm.float64,
            ),
            "Edge barycentric gradients should match the affine edge formula",
        )

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_bc_to_point_via_schema_behind_user_view(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_two_segment_view()
        ctx = segment_view.context()
        bcs = (
            bm.asarray(
                [
                    [1.0, 0.0],
                    [0.25, 0.75],
                ],
                dtype=bm.float64,
            ),
        )

        points = segment_view.schema.bc_to_point(ctx, bcs, None)

        _assert_shape(points, (2, 2, 2), "bc_to_point maps two barycentric samples for each edge entity")
        _assert_allclose(
            points,
            bm.asarray(
                [
                    [[0.0, 0.0], [0.75, 0.0]],
                    [[2.0, 2.0], [2.0, 4.25]],
                ],
                dtype=bm.float64,
            ),
            "Edge barycentric coordinates should map to physical points",
        )

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_quadrature_formula_via_schema_behind_user_view(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_segment_view()

        qf = segment_view.schema.quadrature_formula(2, qtype="legendre")
        bcs, weights = qf.get_quadrature_points_and_weights()

        _assert_shape(bcs, (2, 2), "Edge order-2 quadrature has two barycentric points")
        _assert_shape(weights, (2,), "Edge order-2 quadrature has two weights")

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_grad_shape_function_and_jacobi_matrix_follow_b_u_x_convention(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_two_segment_view()
        ctx = segment_view.context()
        bcs = (
            bm.asarray(
                [
                    [0.75, 0.25],
                    [0.25, 0.75],
                ],
                dtype=bm.float64,
            ),
        )

        grad_b = segment_view.schema.grad_shape_function_barycentric(bcs, (1,))
        grad_u = segment_view.schema.grad_shape_function_reference(bcs, (1,))
        jacobi = segment_view.schema.jacobi_matrix(ctx, bcs, None)
        grad_x = segment_view.grad_shape_function(bcs, p=1, variables="x")

        _assert_shape(grad_b, (2, 2, 2), "Barycentric edge shape gradients are [NQ, num_shape, num_bc]")
        _assert_allclose(
            grad_b,
            bm.asarray(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
                dtype=bm.float64,
            ),
            "For p=1, edge shape functions are lambda0 and lambda1, so their barycentric gradients are constant",
        )

        _assert_shape(grad_u, (2, 2, 1), "Reference edge shape gradients are [NQ, num_shape, ref_dim]")
        _assert_allclose(
            grad_u,
            bm.asarray(
                [
                    [[-1.0], [1.0]],
                    [[-1.0], [1.0]],
                ],
                dtype=bm.float64,
            ),
            "Reference coordinate u maps to lambda = (1-u, u), so dphi/du is [-1, 1]",
        )

        _assert_shape(jacobi, (2, 2, 2, 1), "Edge Jacobi matrices are [entity_count, NQ, GD, ref_dim]")
        _assert_allclose(
            jacobi,
            bm.asarray(
                [
                    [[[1.0], [0.0]], [[1.0], [0.0]]],
                    [[[0.0], [3.0]], [[0.0], [3.0]]],
                ],
                dtype=bm.float64,
            ),
            "Edge Jacobi matrix should equal the physical tangent vector for the affine reference-to-physical map",
        )

        _assert_shape(grad_x, (2, 2, 2, 2), "Cartesian edge shape gradients are [entity_count, NQ, num_shape, GD]")
        _assert_allclose(
            grad_x,
            bm.asarray(
                [
                    [[[-1.0, 0.0], [1.0, 0.0]], [[-1.0, 0.0], [1.0, 0.0]]],
                    [[[0.0, -1.0 / 3.0], [0.0, 1.0 / 3.0]], [[0.0, -1.0 / 3.0], [0.0, 1.0 / 3.0]]],
                ],
                dtype=bm.float64,
            ),
            "Cartesian gradients should scale the reference gradients by the inverse metric along each physical edge",
        )

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_grad_shape_function_user_api_supports_b_u_x_variables(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_two_segment_view()
        bcs = bm.asarray([[0.5, 0.5]], dtype=bm.float64)

        grad_b = segment_view.grad_shape_function(bcs, p=1, variables="b")
        grad_u = segment_view.grad_shape_function(bcs, p=1, variables="u")
        grad_x = segment_view.grad_shape_function(bcs, p=1, variables="x")

        _assert_shape(grad_b, (1, 2, 2), "User API should expose barycentric gradients for edges")
        _assert_shape(grad_u, (1, 2, 1), "User API should expose reference gradients for edges")
        _assert_shape(grad_x, (2, 1, 2, 2), "User API should expose cartesian gradients for each edge entity")

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_normal_and_tangent_through_user_view(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_segment_view()

        tangent = segment_view.tangent()
        normal = segment_view.normal()

        _assert_shape(tangent, (segment_view.size(), 1, segment_view.geo_dimension()), "Handoff rule: tangent shape is [entity_count, T, G]")
        _assert_allclose(
            tangent,
            bm.asarray([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 1.0]], [[0.0, 1.0]]], dtype=bm.float64),
            "Edge tangent is the non-unit endpoint difference x1 - x0",
        )
        _assert_shape(normal, (segment_view.size(), 1, segment_view.geo_dimension()), "2D edge normal has one direction")
        _assert_allclose(
            normal,
            bm.asarray([[[0.0, -1.0]], [[1.0, -0.0]], [[1.0, 1.0]], [[1.0, -0.0]]], dtype=bm.float64),
            "Current 2D edge normal follows [dy, -dx]",
        )

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_scalar_index_keeps_single_entity_axis(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_two_segment_view()

        _assert_allclose(
            segment_view.barycenter(index=1),
            bm.asarray([[2.0, 3.5]], dtype=bm.float64),
            "Scalar index should preserve one edge entity axis in barycenter",
        )
        _assert_allclose(
            segment_view.measure(index=1),
            bm.asarray([3.0], dtype=bm.float64),
            "Scalar index should preserve one edge entity axis in measure",
        )
        _assert_shape(segment_view.grad_lambda(index=1), (1, 2, 2), "Scalar index should preserve grad_lambda entity axis")
        _assert_shape(segment_view.tangent(index=1), (1, 1, 2), "Scalar index should preserve tangent entity axis")

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_3d_normal_returns_two_orthogonal_directions(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_3d_segment_view()

        tangent = segment_view.tangent()
        normal = segment_view.normal()

        _assert_shape(normal, (segment_view.size(), 2, 3), "3D edge normal should have G - T = 2 directions")
        dot_n0_t = bm.sum(normal[:, 0, :] * tangent[:, 0, :], axis=1)
        dot_n1_t = bm.sum(normal[:, 1, :] * tangent[:, 0, :], axis=1)
        dot_n0_n1 = bm.sum(normal[:, 0, :] * normal[:, 1, :], axis=1)
        _assert_allclose(
            dot_n0_t,
            bm.zeros((segment_view.size(),), dtype=bm.float64),
            "First 3D edge normal direction must be orthogonal to tangent",
        )
        _assert_allclose(
            dot_n1_t,
            bm.zeros((segment_view.size(),), dtype=bm.float64),
            "Second 3D edge normal direction must be orthogonal to tangent",
        )
        _assert_allclose(
            dot_n0_n1,
            bm.zeros((segment_view.size(),), dtype=bm.float64),
            "The two 3D edge normal directions must be mutually orthogonal",
        )
        assert bm.all(bm.sum(normal[:, 0, :] * normal[:, 0, :], axis=1) > 0)
        assert bm.all(bm.sum(normal[:, 1, :] * normal[:, 1, :], axis=1) > 0)

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_degenerate_edge_normal_raises_value_error(self, backend):
        bm.set_backend(backend)
        _, segment_view = _build_degenerate_segment_view()

        with pytest.raises(ValueError):
            segment_view.normal()


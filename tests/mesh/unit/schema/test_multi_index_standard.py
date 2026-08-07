import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import (
    PointSchema,
    SegmentSchema,
    TriangleSchema,
    QuadrilateralSchema,
    TetrahedronSchema,
    HexahedronSchema,
    PrismSchema,
    PyramidSchema,
)


def to_numpy(value):
    return bm.to_numpy(value)


def test_simplex_multi_index_shape_is_num_points_by_num_vertices():
    assert PointSchema.multi_index((3,)).shape == (1, 1)
    np.testing.assert_array_equal(to_numpy(PointSchema.multi_index((3,))), [[3]])

    np.testing.assert_array_equal(to_numpy(SegmentSchema.multi_index((2,))), [[2, 0], [1, 1], [0, 2]])
    assert SegmentSchema.multi_index((2,)).shape == (3, 2)

    tri = TriangleSchema.multi_index((2,))
    assert tri.shape == (6, 3)
    np.testing.assert_array_equal(to_numpy(bm.sum(tri, axis=1)), np.full(6, 2))

    tet = TetrahedronSchema.multi_index((2,))
    assert tet.shape == (10, 4)
    np.testing.assert_array_equal(to_numpy(bm.sum(tet, axis=1)), np.full(10, 2))


def test_quadrilateral_multi_index_uses_tensor_product_vertex_order():
    mi = QuadrilateralSchema.multi_index((1, 2))
    expected = np.array([
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 2, 0],
        [0, 0, 0, 2],
    ], dtype=np.int32)
    assert mi.shape == (6, 4)
    np.testing.assert_array_equal(to_numpy(mi), expected)
    assert QuadrilateralSchema.num_multi_index((1, 2)) == 6


def test_hexahedron_multi_index_uses_bottom_then_top_vertex_order():
    mi = HexahedronSchema.multi_index((1, 1, 1))
    expected = np.eye(8, dtype=np.int32)
    assert mi.shape == (8, 8)
    np.testing.assert_array_equal(to_numpy(mi), expected)
    assert HexahedronSchema.num_multi_index((1, 1, 1)) == 8


def test_prism_multi_index_is_interval_times_triangle_with_triangle_fastest():
    mi = PrismSchema.multi_index((1, 2))
    expected = np.array([
        [2, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2],
    ], dtype=np.int32)
    assert mi.shape == (9, 6)
    np.testing.assert_array_equal(to_numpy(mi), expected)


def test_pyramid_multi_index_exists_with_vertex_count_columns_for_linear_order():
    mi = PyramidSchema.multi_index((1,))
    assert mi.shape[1] == 5
    assert mi.shape[0] >= 5
    np.testing.assert_array_equal(to_numpy(mi[:5]), np.eye(5, dtype=np.int32))

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fealpy.backend import bm
from fealpy.mesher.box import Box1d, Box2d, Box3d


def _np(value):
    return bm.to_numpy(value)


def test_box1d_initialize_cache_and_connectivity():
    box = Box1d(box=[-1.0, 1.0], nx=4)
    node, cell = box.initialize()

    expected_node = np.array(
        [[-1.0], [-0.5], [0.0], [0.5], [1.0]],
        dtype=np.float64,
    )
    expected_cell = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        dtype=np.int32,
    )

    np.testing.assert_allclose(_np(node), expected_node)
    np.testing.assert_array_equal(_np(cell), expected_cell)

    node_cached, cell_cached = box.initialize()
    assert node is node_cached
    assert cell is cell_cached

    box.clear()
    node_new, cell_new = box.initialize()
    assert node_new is not node
    assert cell_new is not cell


def test_box1d_segmentize_mesh():
    mesh = Box1d(box=[0.0, 1.0], nx=2).segmentize()

    expected_positions = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)
    np.testing.assert_allclose(_np(mesh.block.positions), expected_positions)

    expected_edge = np.array([[0, 1], [1, 2]], dtype=np.int32)
    np.testing.assert_array_equal(_np(mesh.Entity("edge").indices), expected_edge)

    assert mesh.Entity("edge").size() == 2
    assert mesh.Entity("node").size() == 3

    node_indices = _np(mesh.Entity("node").indices).reshape(-1)
    np.testing.assert_array_equal(node_indices, np.array([0, 1, 2], dtype=np.int32))


def test_box2d_initialize_and_quadrangulate():
    box = Box2d(box=[0.0, 1.0, 0.0, 1.0], nx=1, ny=1)
    node, cell = box.initialize()

    expected_node = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    expected_cell = np.array([[0, 2, 1, 3]], dtype=np.int32)

    np.testing.assert_allclose(_np(node), expected_node)
    np.testing.assert_array_equal(_np(cell), expected_cell)

    mesh = box.quadrangulate()
    np.testing.assert_array_equal(_np(mesh.Entity("quad").indices), expected_cell)
    assert mesh.Entity("cell").size() == 1
    assert mesh.Entity("edge").size() == 4
    assert mesh.Entity("node").size() == 4


def test_box2d_triangulate():
    mesh = Box2d(nx=1, ny=1).triangulate()
    tri = _np(mesh.Entity("tri").indices)

    expected_tri = np.array([[0, 2, 3], [0, 3, 1]], dtype=np.int32)
    np.testing.assert_array_equal(tri, expected_tri)

    assert mesh.Entity("cell").size() == 2
    assert mesh.Entity("edge").size() == 5


def test_box3d_hexahedralize():
    box = Box3d(nx=1, ny=1, nz=1)
    node, cell = box.initialize()

    expected_node = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected_cell = np.array([[0, 4, 2, 6, 1, 5, 3, 7]], dtype=np.int32)

    np.testing.assert_allclose(_np(node), expected_node)
    np.testing.assert_array_equal(_np(cell), expected_cell)

    mesh = box.hexahedralize()
    np.testing.assert_array_equal(_np(mesh.Entity("hex").indices), expected_cell)

    assert mesh.Entity("cell").size() == 1
    assert mesh.Entity("face").size() == 6
    assert mesh.Entity("edge").size() == 12
    assert mesh.Entity("node").size() == 8


@pytest.mark.parametrize(
    "method_name, Entity_name, expected_cells",
    [
        (
            "tetrahedralize",
            "tet",
            np.array(
                [
                    [0, 4, 2, 3],
                    [0, 5, 4, 3],
                    [0, 1, 5, 3],
                    [2, 4, 6, 7],
                    [4, 5, 7, 3],
                    [2, 7, 3, 4],
                ],
                dtype=np.int32,
            ),
        ),
        (
            "prismatize",
            "prism",
            np.array(
                [
                    [0, 4, 2, 1, 5, 3],
                    [2, 4, 6, 3, 5, 7],
                ],
                dtype=np.int32,
            ),
        ),
        (
            "pyramidalize",
            "pyramid",
            np.array(
                [
                    [0, 4, 2, 6, 3],
                    [0, 4, 1, 5, 3],
                    [0, 2, 1, 7, 3],
                ],
                dtype=np.int32,
            ),
        ),
    ],
    ids=["tetrahedralize", "prismatize", "pyramidalize"],
)
def test_box3d_decompositions(method_name, Entity_name, expected_cells):
    box = Box3d(nx=1, ny=1, nz=1)
    if not hasattr(box, method_name):
        pytest.skip(f"Box3d.{method_name} is not implemented")
    mesh = getattr(box, method_name)()

    actual_cells = _np(mesh.Entity(Entity_name).indices)
    np.testing.assert_array_equal(actual_cells, expected_cells)
    assert mesh.Entity("cell").size() == expected_cells.shape[0]

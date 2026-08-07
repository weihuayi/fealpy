import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh.storage import EntitySector, MeshBlock, Relation
from fealpy.mesh.topology.builder import TopologyBuilder
from fealpy.mesh.view import Mesh


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    return bm.to_numpy(value)


def make_two_triangle_fealpy_mesh():
    points = bm.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=bm.float64,
    )
    cells = bm.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=bm.int32,
    )
    block = MeshBlock(positions=points)
    block.add_sector(EntitySector(schema_name="tri", indices=cells), root=True)
    TopologyBuilder.construct(block)
    return Mesh(block).fealpy_api()


def test_relation_unique_homogeneous_returns_boundary_columns_and_local_positions():
    relation = Relation(
        src_name="cell",
        tgt_name="face",
        tgt_indices=bm.tensor([[3, 5, 7], [2, 4, 6]], dtype=bm.int32),
    )

    unique = relation.unique
    lidx = relation.local_index

    np.testing.assert_array_equal(to_numpy(unique.first), np.array([3, 2], dtype=np.int32))
    np.testing.assert_array_equal(to_numpy(unique.last), np.array([7, 6], dtype=np.int32))
    np.testing.assert_array_equal(to_numpy(lidx.floc), np.array([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(to_numpy(lidx.lloc), np.array([2, 2], dtype=np.int32))


def test_relation_unique_heterogeneous_returns_first_last_per_source_with_local_positions():
    relation = Relation(
        src_name="face",
        tgt_name="cell",
        src_indices=bm.tensor([2, 1, 0, 4, 2, 3], dtype=bm.int32),
        tgt_indices=bm.tensor([0, 0, 0, 1, 1, 1], dtype=bm.int32),
    )

    unique = relation.unique
    lidx = relation.local_index

    np.testing.assert_array_equal(to_numpy(unique.first), np.array([0, 0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(to_numpy(unique.last), np.array([0, 0, 1, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(to_numpy(lidx.floc), np.array([2, 1, 0, 2, 0], dtype=np.int32))
    np.testing.assert_array_equal(to_numpy(lidx.lloc), np.array([2, 1, 1, 2, 0], dtype=np.int32))


def test_face_to_cell_returns_left_right_cells_and_local_face_indices():
    mesh = make_two_triangle_fealpy_mesh()

    face_to_cell = mesh.face_to_cell()

    expected = np.array(
        [
            [0, 0, 2, 2],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 2, 2],
            [1, 1, 0, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(to_numpy(face_to_cell), expected)

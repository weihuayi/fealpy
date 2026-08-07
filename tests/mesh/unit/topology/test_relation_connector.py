import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh.storage import MeshBlock, EntitySector
from fealpy.mesh.topology.builder import TopologyBuilder, TopRelationConnector


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    return bm.to_numpy(value)


def test_topology_relation_connector_connects_existing_lower_sector():
    pts = bm.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=bm.float64)
    tri = bm.tensor([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=bm.int32)
    block = MeshBlock(positions=pts)
    block.add_sector(EntitySector(schema_name="tri", indices=tri), root=True)
    TopologyBuilder.construct(block)

    old_segment = block.get_sector("segment").indices.copy()
    old_point = block.get_sector("point").indices.copy()
    block.relations.pop(("segment", "point"), None)

    relation = TopRelationConnector.connect(block, "segment", "point")

    np.testing.assert_array_equal(
        to_numpy(relation.tgt_indices),
        to_numpy(old_segment),
    )
    np.testing.assert_array_equal(
        to_numpy(block.get_sector("segment").indices),
        to_numpy(old_segment),
    )
    np.testing.assert_array_equal(
        to_numpy(block.get_sector("point").indices),
        to_numpy(old_point),
    )

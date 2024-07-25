
import ipdb
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh

def test_triangle_mesh_init():
    node = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2]], dtype=bm.int32)

    mesh = TriangleMesh(node, cell)

    assert mesh.node.shape == (3, 2)
    assert mesh.cell.shape == (1, 3)

    assert mesh.number_of_nodes() == 3
    assert mesh.number_of_cells() == 1




if __name__ == "__main__":
    test_triangle_mesh_init()

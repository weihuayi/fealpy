from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import QuadrangleMesh
# from fealpy.mesh import QuadrangleMesh

# bm.set_backend('pytorch')

def test_quad_mesh_init():
    node = bm.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)

    mesh = QuadrangleMesh(node, cell)

    assert mesh.node.shape == (4, 2)
    assert mesh.cell.shape == (1, 4)

    assert mesh.number_of_nodes() == 4
    assert mesh.number_of_cells() == 1

def test_quad_mesh_entity_measure():
    node = bm.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)

    mesh = QuadrangleMesh(node, cell)

    assert mesh.entity_measure(0) == bm.tensor([0.0], dtype=bm.float64)
    assert all(mesh.entity_measure(1) == bm.tensor([1.0, 1.0, 1.0, 1.0], dtype=bm.float64))
    assert all(mesh.entity_measure('cell') == bm.tensor([1.0], dtype=bm.float64))


if __name__ == '__main__':
    test_quad_mesh_init()
    test_quad_mesh_entity_measure()
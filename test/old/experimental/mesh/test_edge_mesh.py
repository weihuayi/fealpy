from fealpy.backend import backend_manager as bm
from fealpy.mesh.edge_mesh import EdgeMesh
from fealpy.quadrature import GaussLegendreQuadrature
import pytest

def test_init():
    # Define sample node and cell arrays
    node = bm.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=bm.int_)

    # Create EdgeMesh object
    edge_mesh = edge_mesh(node, cell)

    # Test __init__ method
    assert edge_mesh.node is node
    assert edge_mesh.itype == cell.dtype
    assert edge_mesh.ftype == node.dtype
    assert edge_mesh.meshtype == 'edge'
    assert edge_mesh.NN == node.shape[0]
    assert edge_mesh.GD == node.shape[1]
    assert edge_mesh.nodedata == {}
    assert edge_mesh.celldata == {}

    # Test geo_dimension method
    assert edge_mesh.geo_dimension() == 2

    # Test top_dimension method
    assert edge_mesh.top_dimension() == 1


def test_add_plot():
    import matplotlib.pyplot as plt

    mesh = EdgeMesh.from_tower()
    mesh.add_plot(plt)

def test_quadrature_formula():
    node = bm.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=bm.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    q = 4
    inte = edge_mesh.quadrature_formula(q)
    a,b = inte.get_quadrature_points_and_weights()
    print(a)
    print("\n")
    print(b)

def test_entity_measure():
    node = bm.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=bm.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    a = edge_mesh.entity_measure()
    print(a)

def test_grad_lambda():
    node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bm.float64)
    cell = bm.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=bm.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    val = edge_mesh.grad_lambda()
    print(val)


def test_number_of_local_ipoints():
    node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bm.float64)
    cell = bm.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=bm.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    a = edge_mesh.number_of_local_ipoints(p=1)
    print(a)


def test_number_of_global_ipoints():
    node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bm.float64)
    cell = bm.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=bm.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    val = edge_mesh.number_of_global_ipoints(p=3)
    print(val)

def test_interpolation_points():
    node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bm.float64)
    cell = bm.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=bm.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    a = edge_mesh.interpolation_points(p=2)
    print(a)

def test_cell_normal():
    node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bm.float64)
    cell = bm.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=bm.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    val = edge_mesh.cell_normal()
    print(val)

def test_edge_length():
    node = bm.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bm.float64)
    cell = bm.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=bm.int_)
    mesh = EdgeMesh(node,cell)
    edge_length = mesh.edge_length()
    print(edge_length)
    print(edge_length.dtype)

if __name__ == "__main__":
    # test_quadrature_formula()
    # test_entity_measure()
    # test_grad_lambda()
    # test_number_of_local_ipoints()
    # test_number_of_global_ipoints()
    # test_interpolation_points()
    # test_cell_normal()
    test_edge_length()

    

import numpy as np
from fealpy.mesh.edge_mesh import EdgeMesh
from fealpy.quadrature import GaussLegendreQuadrature
import pytest

def test_init():
    # Define sample node and cell arrays
    node = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    cell = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)

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

def test_grad_function():
    node = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    cell = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    bc = np.array([[0.2,0.8]],dtype=np.float_)
    va = edge_mesh.grad_shape_function(bc,p=1)
    print(va)

def test_entity_measure():
    node = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    cell = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    a = edge_mesh.entity_measure()
    print(a)

def test_grad_lambda():
    node = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    cell = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    val = edge_mesh.grad_lambda()
    print(val)

def test_interpolation_points():
    node = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    cell = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    a = edge_mesh.interpolation_points(p=2)
    print(a)

def test_cell_normal():
    node = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    cell = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int_)

    # Create EdgeMesh object
    edge_mesh = EdgeMesh(node, cell)
    val = edge_mesh.cell_normal()
    print(val)

'''
if __name__ == "__main__":
    # test_add_plot()
    # test_grad_function()
    # test_entity_measure()
    # test_grad_lambda()
    # test_interpolation_points()
    # test_cell_normal()
'''
node = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
cell = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int_)

# Create EdgeMesh object
mesh = EdgeMesh(node, cell)
a = mesh.cell_normal()
print((a,))

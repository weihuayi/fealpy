import numpy as np
from fealpy.mesh import EdgeMesh
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
    #plt.show()


if __name__ == "__main__":
    test_add_plot()
    

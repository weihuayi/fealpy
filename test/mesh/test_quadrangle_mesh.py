import numpy as np
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh import QuadrangleMesh
from fealpy.quadrature import TensorProductQuadrature, GaussLegendreQuadrature

def test_QuadrangleMesh_constructor():
    node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    cell = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    quad_mesh = QuadrangleMesh(node, cell)

    assert quad_mesh.node.shape == (4, 2)
    assert quad_mesh.ds.NN == 4
    assert quad_mesh.meshtype == 'quad'
    assert quad_mesh.p == 1
    assert quad_mesh.itype == np.uint64
    assert quad_mesh.ftype == np.float64
    assert isinstance(quad_mesh.celldata, dict)
    assert isinstance(quad_mesh.nodedata, dict)
    assert isinstance(quad_mesh.edgedata, dict)
    assert isinstance(quad_mesh.meshdata, dict)

def test_QuadrangleMesh_integrator():
    node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    cell = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    quad_mesh = QuadrangleMesh(node, cell)
    integrator_cell = quad_mesh.integrator(3, etype='cell')
    integrator_edge = quad_mesh.integrator(3, etype='edge')

    assert isinstance(integrator_cell, TensorProductQuadrature)
    assert isinstance(integrator_edge, GaussLegendreQuadrature)

@pytest.mark.parametrize("vertices, h", [
    ([(0, 0), (1, 0), (1, 1), (0, 1)], 0.1),
    ([(0, 0), (1, 0), (1, 1), (0, 1)], 0.5),
    ([(0, 0), (1, 0), (0.5, 1)], 0.1),
])
def test_QuadrangleMesh_from_polygon_gmsh(vertices, h):
    quad_mesh = QuadrangleMesh.from_polygon_gmsh(vertices, h)

    #fig = plt.figure()
    #axes = fig.add_subplot(111)
    #quad_mesh.add_plot(axes)
    #plt.show()


    assert isinstance(quad_mesh, QuadrangleMesh)
    assert quad_mesh.node.shape[1] == 2
    assert quad_mesh.ds.NN >= len(vertices)
    assert quad_mesh.meshtype == 'quad'
    assert quad_mesh.p == 1
    assert quad_mesh.itype == np.uint64
    assert quad_mesh.ftype == np.float64


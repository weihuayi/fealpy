import numpy as np
import ipdb
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh import QuadrangleMesh
from fealpy.quadrature import TensorProductQuadrature, GaussLegendreQuadrature

def test_quadrangle_mesh_constructor():
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

def test_quadrangle_mesh_integrator():
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
def test_quadrangle_mesh_from_polygon_gmsh(vertices, h):
    quad_mesh = QuadrangleMesh.from_polygon_gmsh(vertices, h)

    assert isinstance(quad_mesh, QuadrangleMesh)
    assert quad_mesh.node.shape[1] == 2
    assert quad_mesh.ds.NN >= len(vertices)
    assert quad_mesh.meshtype == 'quad'
    assert quad_mesh.p == 1
    assert quad_mesh.itype == np.uint64
    assert quad_mesh.ftype == np.float64


def test_quadrangle_mesh_interpolate():
    node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
    mesh = QuadrangleMesh(node, cell)
    mesh.uniform_refine(1)
    ips = mesh.interpolation_points(4)
    c2p = mesh.cell_to_ipoint(4)
    #fig, axes = plt.subplots()
    #mesh.add_plot(axes)
    #mesh.find_node(axes, node=ips, showindex=True)
    #plt.show()

def test_quadrangle_mesh_shape_function():
    mesh = QuadrangleMesh.from_one_quadrangle()
    bcs, ws = mesh.integrator(3).get_quadrature_points_and_weights()
    phi = mesh.shape_function(bcs, p=1)
    gphi = mesh.grad_shape_function(bcs, p=1)

    phi = mesh.shape_function(bcs, p=2)
    gphi = mesh.grad_shape_function(bcs, p=2)

    J = mesh.jacobi_matrix(bcs)
    print(np.linalg.det(J))



if __name__ == "__main__":
    test_quadrangle_mesh_shape_function()


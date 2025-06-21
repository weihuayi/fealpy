import numpy as np
import ipdb
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh.polygon_mesh import PolygonMesh 
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import ConformingScalarVESpace2d 

def test_polygon_mesh_constructor():
    node = np.array([
        (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
        (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
            dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)

    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    plt.show()

def test_polygon_mesh_interpolation_points_4():
    node = np.array([
        (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
        (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
            dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)

    gdof = mesh.number_of_global_ipoints(p)
    ips = mesh.interpolation_points(p)

    c2p = mesh.cell_to_ipoint(p)
    result = [
            np.array([ 0, 12, 13, 14,  3, 27, 28, 29,  4, 15, 16, 17, 48, 49, 50, 51, 52, 53]), 
            np.array([ 4, 21, 22, 23,  1,  9, 10, 11,  0, 17, 16, 15, 54, 55, 56, 57, 58, 59]), 
            np.array([ 1, 23, 22, 21,  4, 33, 34, 35,  5, 24, 25, 26,  2, 18, 19, 20, 60, 61, 62, 63, 64, 65]), 
            array([ 3, 30, 31, 32,  6, 42, 43, 44,  7, 36, 37, 38,  4, 29, 28, 27, 66, 67, 68, 69, 70, 71]), 
            array([ 4, 38, 37, 36,  7, 45, 46, 47,  8, 39, 40, 41,  5, 35, 34, 33, 72, 73, 74, 75, 76, 77])
            ]
    for a0, a1 in zip(c2p, result):
        np.testing.assert_equal(a0, a1) 

    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    mesh.find_node(axes, node=ips, showindex=True)
    plt.show()

@pytest.mark.parametrize('meshtype', ['equ', 'iso'])
def test_from_one_triangle(meshtype): 
    mesh = PolygonMesh.from_one_triangle(meshtype=meshtype)
    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True)
    mesh.find_edge(axes, showindex=True)
    plt.show()

def test_from_one():
    #mesh = PolygonMesh.from_one_square()
    #mesh = PolygonMesh.from_one_pentagon() 
    #mesh = PolygonMesh.from_one_hexagon()
    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True)
    mesh.find_edge(axes, showindex=True)
    plt.show()
def test_from_triangle_mesh_by_dual():
    mesh = TriangleMesh.from_one_triangle()
    mesh.uniform_refine()
    mesh = PolygonMesh.from_triangle_mesh_by_dual(mesh)
 
    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True)
    mesh.find_edge(axes, showindex=True)
    plt.show()
def test_integral():
    nx = 20
    ny = 20
    domain = [0, 1, 0, 1]
    tmesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    mesh = PolygonMesh.from_triangle_mesh_by_dual(tmesh)
    p = 2
    space =  ConformingScalarVESpace2d(mesh, p=p)
    phi = space.smspace.basis
    def f(p, index):
        x = p[...,0]
        y = p[...,1]
        val = x**2+y**2
        return val
    a = mesh.integral(f, q=5, celltype=False)
    np.testing.assert_allclose(a,2/3,atol=1e-16)
    return a 

if __name__ == "__main__":
    #test_polygon_mesh_constructor()
    #test_polygon_mesh_interpolation_points(4)
    #test_from_one_triangle('iso')
    #test_from_one()
    #test_from_triangle_mesh_by_dual()
    test_integral()

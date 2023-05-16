import numpy as np
import ipdb
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh.polygon_mesh import PolygonMesh 

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

def test_polygon_mesh_interpolation_points(p):
    node = np.array([
        (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
        (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
            dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)

    gdof = mesh.number_of_global_ipoints(p)

    ips = mesh.interpolation_points(p)

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


if __name__ == "__main__":
    #test_polygon_mesh_constructor()
    test_polygon_mesh_interpolation_points(4)
    #test_from_one_triangle('iso')
    #test_from_one()


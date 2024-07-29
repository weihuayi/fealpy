import ipdb
import pytest
import matplotlib.pyplot as plt
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh
from fealpy.mesh import TriangleMesh as TMesh

def test_triangle_mesh_init():
    node = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2]], dtype=bm.int32)

    mesh = TriangleMesh(node, cell)

    assert mesh.node.shape == (3, 2)
    assert mesh.cell.shape == (1, 3)

    assert mesh.number_of_nodes() == 3
    assert mesh.number_of_cells() == 1

def test_triangle_mesh_uniform_refine():

    mesh = TriangleMesh.from_one_triangle(meshtype='iso')
    mesh.uniform_refine(n=1)

    assert mesh.node.shape == (6, 2)
    assert mesh.cell.shape == (4, 3)

    assert mesh.number_of_nodes() == 6
    assert mesh.number_of_cells() == 4

def test_triangle_mesh_from_sphere_surface():
    mesh = TriangleMesh.from_unit_sphere_surface()
    assert mesh.node.shape == (12, 3)
    assert mesh.cell.shape == (20, 3)

    assert mesh.number_of_nodes() == 12 
    assert mesh.number_of_cells() == 20 

def test_triangle_mesh_from_box():
    mesh = TriangleMesh.from_box(nx=2, ny=2)
    
    assert mesh.node.shape == (9, 2)
    assert mesh.cell.shape == (8, 3)

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    
    if False:
        tmesh = TMesh(node, cell)
        fig = plt.figure()
        axes = fig.gca()
        tmesh.add_plot(axes)
        tmesh.find_node(axes, showindex=True)
        tmesh.find_cell(axes, showindex=True)
        plt.show()

def test_triangle_mesh_from_torus_surface():
    mesh = TriangleMesh.from_torus_surface(R=2, r=1, nu=2, nv=2)
    
    assert mesh.node.shape == (4, 3)
    assert mesh.cell.shape == (8, 3)

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    
    #if False:
    tmesh = TMesh(node, cell)
    fig = plt.figure()
    axes = fig.gca(projection='3d')
    tmesh.add_plot(axes)
    tmesh.find_node(axes, showindex=True)
    tmesh.find_cell(axes, showindex=True)
    plt.show()

if __name__ == "__main__":
    test_triangle_mesh_init()
    test_triangle_mesh_uniform_refine()
    test_triangle_mesh_from_sphere_surface()
    test_triangle_mesh_from_box()
    test_triangle_mesh_from_torus_surface()

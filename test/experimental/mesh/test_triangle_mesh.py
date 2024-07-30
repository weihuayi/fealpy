import ipdb
import pytest
import numpy as np
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

def test_triangle_mesh_entity_measure():
    node = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
    cell = bm.tensor([[0, 1, 2]], dtype=bm.int32)

    mesh = TriangleMesh(node, cell)
    assert mesh.entity_measure(0) == bm.tensor([0.0], dtype=bm.float64)
    assert all(mesh.entity_measure(1) == bm.tensor([1.0, 1.0, np.sqrt(2)], dtype=bm.float64))
    assert all(mesh.entity_measure('cell') == bm.tensor([0.5], dtype=bm.float64))

def test_quadrature_formula():
    q = 3
    mesh = TriangleMesh.from_one_triangle(meshtype='iso')
    quad0 = mesh.quadrature_formula(q=q)

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    tmesh = TMesh(node, cell)
    quad1 = tmesh.integrator(q=q)

def test_grad_lambda():
    mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=16, ny=16)
    flag = mesh.boundary_cell_flag()
    NC = bm.sum(flag)
    val = mesh.grad_lambda(index=flag)
    assert val.shape == (NC, 3, 2)

def test_interpolation_points():
    mesh = TriangleMesh.from_box(nx=2, ny=2)
    
    gdof0 = mesh.number_of_global_ipoints(p=1)
    inode0 = mesh.interpolation_points(p=1)
    
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    tmesh = TMesh(node, cell)
    inode1 = tmesh.interpolation_points(p=1)
    gdof1 = tmesh.number_of_global_ipoints(p=1)
    
    assert np.array_equal(inode0, inode1)
    assert inode0.shape == inode1.shape

def test_cell_to_ipoint():
    mesh = TriangleMesh.from_box(nx=2, ny=2)

    icell0 = mesh.cell_to_ipoint(p=1)
    
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    tmesh = TMesh(node, cell)
    icell1 = tmesh.cell_to_ipoint(p=1)

    assert np.array_equal(icell0, icell1)
    assert icell0.shape == icell1.shape 

def test_triangle_mesh_uniform_refine():

    mesh = TriangleMesh.from_one_triangle(meshtype='iso')
    mesh.uniform_refine(n=1)

    assert mesh.node.shape == (6, 2)
    assert mesh.cell.shape == (4, 3)

    assert mesh.number_of_nodes() == 6
    assert mesh.number_of_cells() == 4

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
    mesh = TriangleMesh.from_torus_surface(R=10, r=2, nu=10, nv=10)
    
    assert mesh.node.shape == (100, 3)
    assert mesh.cell.shape == (200, 3)

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    
    if False:
        tmesh = TMesh.from_torus_surface(node, cell)
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        tmesh.add_plot(axes)
        tmesh.find_node(axes, showindex=True)
        tmesh.find_cell(axes, showindex=True)
        plt.show()

def test_triangle_mesh_from_sphere_surface():
    mesh = TriangleMesh.from_unit_sphere_surface()
    assert mesh.node.shape == (12, 3)
    assert mesh.cell.shape == (20, 3)

    assert mesh.number_of_nodes() == 12 
    assert mesh.number_of_cells() == 20 


if __name__ == "__main__":
    test_triangle_mesh_init()
    test_triangle_mesh_entity_measure()
    test_quadrature_formula()
    #test_grad_lambda()
    #test_interpolation_points()
    #test_cell_to_ipoint()
    #test_triangle_mesh_uniform_refine()
    #test_triangle_mesh_from_box()
    #test_triangle_mesh_from_torus_surface()
    #test_triangle_mesh_from_sphere_surface()

import numpy as np
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.mesh.polygon_mesh import PolygonMesh
from fealpy.functionspace.conforming_vector_ve_space_2d import CVVEDof2d 
import ipdb

def test_dof_3(p):
    tmesh = TriangleMesh.from_one_triangle()
    tmesh.uniform_refine()
    pmesh = PolygonMesh.from_mesh(tmesh) 
    dof = CVVEDof2d(pmesh, p)

    isBdDof = dof.is_boundary_dof()
    result = np.array([True,  True,  True,  True , True,  True,  True , True, True,  True,  True,  True,True,  True,  True,  True , True,  True,  True,True,  True,  True, True,True, True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False, False, False, False])
    np.testing.assert_equal(isBdDof, result)

    gdof = dof.number_of_global_dofs()
    assert gdof==72

    local = dof.number_of_local_dofs()
    result = np.array([24, 24, 24, 24])
    np.testing.assert_equal(local, result)

    edge2dof = dof.edge_to_dof()
    result = np.array([[ 0,  1, 12, 13, 14, 15,  6,  7],
         [ 8 , 9, 16, 17, 18 ,19,  0,  1],
         [ 6,  7, 20, 21, 22, 23,  2,  3],
         [ 2,  3 ,24, 25 ,26, 27 ,10 ,11],
         [ 4,  5 ,28 ,29 ,30 ,31  ,8  ,9],
         [10, 11, 32 ,33 ,34 ,35  ,4  ,5],
         [ 6,  7, 36 ,37, 38 ,39 , 8  ,9],
         [10, 11, 40, 41, 42, 43,  6,  7],
         [ 8,  9, 44, 45, 46, 47, 10, 11]])
    np.testing.assert_equal(edge2dof, result)

    cell2dof = dof.cell_to_dof()
    result = [np.array([ 0,  1, 12, 13, 14, 15,  6,  7, 36, 37, 38, 39,  8,  9, 16, 17, 18,
       19, 48, 49, 50, 51, 52, 53]), np.array([ 6,  7, 20, 21, 22, 23,  2,  3, 24, 25, 26, 27, 10, 11, 40, 41, 42,
       43, 54, 55, 56, 57, 58, 59]), np.array([ 8,  9, 44, 45, 46, 47, 10, 11, 32, 33, 34, 35,  4,  5, 28, 29, 30,
       31, 60, 61, 62, 63, 64, 65]), np.array([11, 10, 47, 46, 45, 44,  9,  8, 39, 38, 37, 36,  7,  6, 43, 42, 41,
       40, 66, 67, 68, 69, 70, 71])]
    for a0, a1 in zip(cell2dof, result):
        np.testing.assert_equal(a0, a1) 

    if True:
        fig, axes = plt.subplots()
        pmesh.add_plot(axes)
        pmesh.find_node(axes, showindex=True)
        pmesh.find_cell(axes, showindex=True)
        pmesh.find_edge(axes, showindex=True)
        plt.show()

def test_interpolation_points_4(p):
    tmesh = TriangleMesh.from_one_triangle()
    tmesh.uniform_refine()
    pmesh = PolygonMesh.from_mesh(tmesh) 

    dof = CVVEDof2d(pmesh, p)
    ips = dof.interpolation_points(scale=0.3)
    #np.testing.assert_allclose(ips[-1], np.array([0.31565566, 0.30271471]),atol=1e-6)

    fig, axes = plt.subplots()
    pmesh.add_plot(axes)
    pmesh.find_node(axes, node=ips, showindex=True)
    pmesh.find_cell(axes, showindex=True)
    pmesh.find_edge(axes, showindex=True)
    plt.show()




if __name__ == "__main__":
    test_dof_3(3)
    test_interpolation_points_4(3)

import numpy as np
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.mesh.polygon_mesh import PolygonMesh
from fealpy.functionspace.non_conforming_scalar_ve_space_2d import NCSVEDof2d



def test_dof_3(p, plot=False):
    node = np.array([
        (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
        (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
    cell = np.array([0, 3, 4, 4, 1, 0, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
            dtype=np.int_)
    cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
    mesh = PolygonMesh(node, cell, cellLocation)

    dof = NCSVEDof2d(mesh, p)
   
    isBdDof = dof.is_boundary_dof()
    result = np.array([ True,  True , True,  True,  True,  True, False, False  ,False,  True,  True,  True,False, False, False,  True,  True,  True, False, False, False,  True,  True,True, False, False, False, False, False, False,  True,  True,  True,  True,  True,True,  True,  True,  True, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False])
    np.testing.assert_equal(isBdDof, result)

    edge2dof = dof.edge_to_dof()
    np.testing.assert_equal(edge2dof[-1], np.array([36, 37, 38]))
    gdof = dof.number_of_global_dofs()
    assert gdof == 54

    ldof = dof.number_of_local_dofs()
    result = np.array([12, 12, 15, 15, 15])
    np.testing.assert_equal(ldof, result)
    cell2dof = dof.cell_to_dof()
    result = [np.array([ 3,  4,  5, 18, 19, 20,  6,  7,  8, 39, 40, 41]), 
            np.array([12, 13, 14,  0,  1,  2,  8,  7,  6, 42, 43, 44]), 
            np.array([14, 13, 12, 24, 25, 26, 15, 16, 17,  9, 10, 11, 45, 46, 47]), 
            np.array([21, 22, 23, 33, 34, 35, 27, 28, 29, 20, 19, 18, 48, 49, 50]), 
            np.array([29, 28, 27, 36, 37, 38, 30, 31, 32, 26, 25, 24, 51, 52, 53])]
    for a0, a1 in zip(cell2dof, result):
        np.testing.assert_equal(a0, a1)
    
    ips = dof.interpolation_points()
    np.testing.assert_allclose(ips[-1], np.array([1.5, 1.67320508]),atol=1e-6)

    if plot:
        fig, axes = plt.subplots()
        mesh.add_plot(axes)
        mesh.find_node(axes, node=ips, showindex=True)
        mesh.find_cell(axes, showindex=True)
        mesh.find_edge(axes, showindex=True)
        plt.show()

if __name__ == "__main__":
    test_dof_3(3, True)

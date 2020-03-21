#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve

from fealpy.functionspace import ReducedDivFreeNonConformingVirtualElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh import PolygonMesh, QuadrangleMesh
from fealpy.pde.poisson_2d import CosCosData

class ReducedDivFreeNonConformingVirtualElementSpace2dTest:

    def __init__(self):
        pass

    def project_test(self, u, p=2, mtype=0, plot=True):
        from fealpy.mesh.simple_mesh_generator import triangle
        if mtype == 0:
            node = np.array([
                (-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float)
            cell = np.array([0, 1, 2, 3], dtype=np.int)
            cellLocation = np.array([0, 4], dtype=np.int)
            mesh = PolygonMesh(node, cell, cellLocation)
        elif mtype == 1:
            node = np.array([
                (-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float)
            cell = np.array([0, 1, 2, 3, 0, 2], dtype=np.int)
            cellLocation = np.array([0, 3, 6], dtype=np.int)
            mesh = PolygonMesh(node, cell, cellLocation)
        elif mtype == 2:
            h = 0.1
            mesh = triangle([-1, 1, -1, 1], h, meshtype='polygon')
        elif mtype == 3:
            node = np.array([
                (-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float)
            cell = np.array([[0, 1, 2, 3]], dtype=np.int)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine()
            mesh = PolygonMesh.from_mesh(mesh)


        cell, cellLocation = mesh.entity('cell')
        edge = mesh.entity('edge')
        cell2edge = mesh.ds.cell_to_edge()
        bc = mesh.entity_barycenter('edge')
        print("edge:", edge)
        print("cell2edge:", cell2edge)
        uspace = ReducedDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        up = uspace.project(u)
        print("up:", up)
        up = uspace.project_to_smspace(up)
        print("ups:", up)

        integralalg = uspace.integralalg
        error = integralalg.L2_error(u, up)
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()
def u2(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.zeros(p.shape, p.dtype)
    val[..., 0] = y**2/4
    val[..., 1] = x**2/4
    return val

def u3(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.zeros(p.shape, p.dtype)
    val[..., 0] = y**3/8
    val[..., 1] = x**3/8
    return val

test = ReducedDivFreeNonConformingVirtualElementSpace2dTest()
test.project_test(u2, p=2, mtype=0, plot=True)
#test.project_test(u3, p=3, mtype=3, plot=False)
#test.stokes_equation_test()

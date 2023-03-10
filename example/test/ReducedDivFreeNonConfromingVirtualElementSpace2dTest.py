#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
import scipy.io as sio

from fealpy.functionspace import ReducedDivFreeNonConformingVirtualElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh import PolygonMesh, QuadrangleMesh
from fealpy.pde.poisson_2d import CosCosData

class ReducedDivFreeNonConformingVirtualElementSpace2dTest:

    def __init__(self):
        pass

    def verify_matrix(self, u, p=2, mtype=0, plot=True):
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

        uspace = ReducedDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        uspace.verify_matrix()
        up = uspace.project(u)
        up = uspace.project_to_smspace(up)

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
            h = 0.025
            mesh = triangle([-1, 1, -1, 1], h, meshtype='polygon')
        elif mtype == 3:
            node = np.array([
                (-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float)
            cell = np.array([[0, 1, 2, 3]], dtype=np.int)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine()
            mesh = PolygonMesh.from_mesh(mesh)
        elif mtype == 4:
            node = np.array([
                (-1, -1), ( 1, -1), ( 1, 0), 
                ( 1,  1), (-1,  1), (-1, 0)], dtype=np.float)
            cell = np.array([0, 1, 2, 5, 2, 3, 4, 5], dtype=np.int)
            cellLocation = np.array([0, 4, 8], dtype=np.int)
            mesh = PolygonMesh(node, cell, cellLocation)


        if True:
            cell, cellLocation = mesh.entity('cell')
            edge = mesh.entity('edge')
            cell2edge = mesh.ds.cell_to_edge()
            bc = mesh.entity_barycenter('edge')
            uspace = ReducedDivFreeNonConformingVirtualElementSpace2d(mesh, p)
            up = uspace.project(u)
            up = uspace.project_to_smspace(up)
            print(up)

            integralalg = uspace.integralalg
            error = integralalg.L2_error(u, up)
            print(error)

            A = uspace.matrix_A()
            P = uspace.matrix_P()

            sio.savemat('A.mat', {"A":A.toarray(), "P":P.toarray()})



        if plot:
            mesh.print()
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def stokes_equation_test(self, p=2, maxit=4, mtype=1):
        from scipy.sparse import bmat
        from fealpy.pde.Stokes_Model_2d import CosSinData, PolyY2X2Data
        from fealpy.mesh.simple_mesh_generator import triangle
        h = 0.4
        #pde = CosSinData()
        pde = PolyY2X2Data()
        error = np.zeros((maxit,), dtype=np.float)
        for i in range(maxit):
            if mtype == 0:
                node = np.array([
                    (-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float)
                cell = np.array([0, 1, 2, 3], dtype=np.int)
                cellLocation = np.array([0, 4], dtype=np.int)
                mesh = PolygonMesh(node, cell, cellLocation)
            else:
                mesh = pde.init_mesh(n=i+2, meshtype='poly') 

            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()
            idof = (p-2)*(p-1)//2

            if True:
                fig = plt.figure()
                axes = fig.gca()
                mesh.add_plot(axes)

            uspace = ReducedDivFreeNonConformingVirtualElementSpace2d(mesh, p)
            pspace = ScaledMonomialSpace2d(mesh, 0)

            isBdDof = uspace.boundary_dof()

            udof = uspace.number_of_global_dofs()
            pdof = pspace.number_of_global_dofs()

            uh = uspace.function()
            ph = pspace.function()
            uspace.set_dirichlet_bc(uh, pde.dirichlet)

            A = uspace.matrix_A()
            P = uspace.matrix_P()
            F = uspace.source_vector(pde.source)

            AA = bmat([[A, P.T], [P, None]], format='csr')
            FF = np.block([F, np.zeros(pdof, dtype=uspace.ftype)])
            x = np.block([uh, ph])
            isBdDof = np.block([isBdDof, isBdDof, np.zeros(NC*idof+pdof, dtype=np.bool_)])

            gdof = udof + pdof
            FF -= AA@x
            bdIdx = np.zeros(gdof, dtype=np.int)
            bdIdx[isBdDof] = 1
            Tbd = spdiags(bdIdx, 0, gdof, gdof)
            T = spdiags(1-bdIdx, 0, gdof, gdof)
            AA = T@AA@T + Tbd
            FF[isBdDof] = x[isBdDof]
            x[:] = spsolve(AA, FF)
            uh[:] = x[:udof]
            ph[:] = x[udof:]

            up = uspace.project_to_smspace(uh)
            integralalg = uspace.integralalg
            error[i] = integralalg.L2_error(pde.velocity, up)
            h /= 2

        print(error)
        print(error[0:-1]/error[1:])
        plt.show()

def u0(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.zeros(p.shape, p.dtype)
    val[..., 0] = 1.0 
    val[..., 1] = 1.0 
    return val

def u1(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.zeros(p.shape, p.dtype)
    val[..., 0] = y/2
    val[..., 1] = x/2
    return val

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

def u4(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.zeros(p.shape, p.dtype)
    val[..., 0] = y**4/16
    val[..., 1] = x**4/16
    return val

test = ReducedDivFreeNonConformingVirtualElementSpace2dTest()

if False:
    test.verify_matrix(u3, p=3, mtype=0, plot=True)

if False:
    test.project_test(u2, p=2, mtype=0, plot=False)

if True:
    test.stokes_equation_test(p=2)

#test.project_test(u3, p=3, mtype=3, plot=False)
#test.stokes_equation_test()

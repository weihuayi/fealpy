#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve

from fealpy.functionspace import StokesDivFreeNonConformingVirtualElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh import PolygonMesh
from fealpy.pde.poisson_2d import CosCosData

class StokesDivFreeNonConformingVirtualElementSpace2dTest:

    def __init__(self, p=2, h=0.2):
        self.pde = CosCosData()

    def test_index1(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        idx = space.index1(p=3)
        print(idx)

    def test_index2(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        idx = space.index2(p=3)
        print(idx)

    def test_matrix(self, p=2):
        """
        node = np.array([
            (-1.0, -1.0),
            ( 0.0, -1.0),
            ( 1.0, -1.0),
            (-1.0, 0.0),
            ( 1.0, 0.0),
            (-1.0, 1.0),
            ( 0.0, 1.0),
            ( 1.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 4, 7, 6, 5, 3], dtype=np.int)
        cellLocation = np.array([0, 8], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        """

        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        print("G:", space.G)
        print("B:", space.B)
        print("R:", space.R)
        print("J:", space.J)
        print("Q:", space.Q)
        print("L:", space.L)
        print("D:", space.D)

    def test_matrix_A(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        A = space.matrix_A()
        print(A)


    def test_matrix_P(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        P = space.matrix_P()
        print(P)

    def test_stokes_equation(self, p=2, maxit=4):
        from scipy.sparse import bmat
        from fealpy.pde.Stokes_Model_2d import CosSinData
        from fealpy.mesh.simple_mesh_generator import triangle
        h = 0.1
        pde = CosSinData()
        domain = pde.domain()
        error = np.zeros((maxit,), dtype=np.float)
        for i in range(maxit):
            mesh = triangle(domain, h, meshtype='polygon')

            if 0:
                fig = plt.figure()
                axes = fig.gca()
                mesh.add_plot(axes)
                mesh.find_cell(axes, index=np.array([51], dtype=np.int))
                plt.show()

            uspace = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
            pspace = ScaledMonomialSpace2d(mesh, p-1)

            """
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
            x = np.block([uh.T.flat, ph])
            isBdDof = np.block([isBdDof, isBdDof, np.zeros(pdof, dtype=np.bool)])

            gdof = 2*udof + pdof
            FF -= AA@x
            bdIdx = np.zeros(gdof, dtype=np.int)
            bdIdx[isBdDof] = 1
            Tbd = spdiags(bdIdx, 0, gdof, gdof)
            T = spdiags(1-bdIdx, 0, gdof, gdof)
            AA = T@AA@T + Tbd
            FF[isBdDof] = x[isBdDof]
            x[:] = spsolve(AA, FF)
            uh[:, 0] = x[:udof]
            uh[:, 1] = x[udof:2*udof]
            ph[:] = x[2*udof:]
            """

            up = uspace.project(pde.velocity)
            up = uspace.project_to_smspace(up)
            integralalg = uspace.integralalg
            error[i] = integralalg.L2_error(pde.velocity, up)
            h /= 2


        print(error)
        print(error[0:-1]/error[1:])

    def project_test(self, p=2):
        from fealpy.pde.Stokes_Model_2d import CosSinData
        from fealpy.mesh.simple_mesh_generator import triangle

        h = 0.1
        pde = CosSinData()
        domain = pde.domain()
        mesh = triangle(domain, h, meshtype='polygon')
        uspace = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        pspace = ScaledMonomialSpace2d(mesh, p-1)
        up = uspace.project(pde.velocity)
        up = uspace.project_to_smspace(up)
        integralalg = uspace.integralalg
        error = integralalg.L2_error(pde.velocity, up)
        print(error)

        if 0:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_cell(axes, index=np.array([51], dtype=np.int))
            plt.show()

test = StokesDivFreeNonConformingVirtualElementSpace2dTest()
#test.test_index1()
#test.test_index2()
#test.test_matrix(p=2)
#test.test_matrix_A()
#test.test_matrix_P()
test.project_test()

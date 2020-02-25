#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse.linalg import spsolve

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.mesh import PrismMesh
from fealpy.functionspace import PrismFiniteElementSpace
from fealpy.pde.poisson_2d import CosCosData
from fealpy.boundarycondition import BoundaryCondition
from fealpy.pde.poisson_3d import CosCosCosData


class PrismFiniteElementSpaceTest():
    def __init__(self, p=2):
        pass

    def one_prism_mesh(self):
        node = np.array([
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1)], dtype=np.float)
        cell = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int)
        return PrismMesh(node, cell)

    def plane_pmesh(self, n=0):
        pde = CosCosData()
        mesh = pde.init_mesh(n=n)
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NN = mesh.number_of_nodes()
        pnode = np.zeros((2*NN, 3), dtype=mesh.ftype)
        pnode[:NN, 0:2] = node
        pnode[NN:, 0:2] = node
        pnode[NN:, 2] = 1
        pcell = np.r_['1', cell, cell + NN]

        pmesh = PrismMesh(pnode, pcell)
        pmesh.uniform_refine(n=n)
        return pmesh

    def test_cell_to_dof(self, plot=False):
        mesh = self.space.mesh
        cell = mesh.entity('cell')
        cell2dof = self.space.cell_to_dof()
        ipoints = self.space.interpolation_points()
        print('cell:', cell)
        print('cell2dof:', cell2dof)
        if plot is True:
            fig = plt.figure()
            axes = Axes3D(fig)
            mesh.add_plot(axes, alpha=0, showedge=True)
            mesh.find_node(axes, showindex=True, color='r')
            mesh.find_node(axes, node=ipoints[cell2dof[0]], showindex=True, color='b')
            plt.show()

    def face_to_dof_test(self, p=3, plot=False):
        mesh = self.one_prism_mesh()
        cell = mesh.entity('cell')
        face = mesh.entity('face')
        space = PrismFiniteElementSpace(mesh, p=p)
        cell2dof = space.cell_to_dof()
        face2cell = mesh.ds.face_to_cell()
        print('cell', cell)
        print('face', face)
        print('face2cell', face2cell)
        print('cell2dof', cell2dof)
        ipoints = space.interpolation_points()
        localFace2dof = space.dof.local_face_to_dof()
        face2dof = space.face_to_dof()
        print(localFace2dof)
        if plot is True:
            fig = plt.figure()
            axes = Axes3D(fig)
            mesh.add_plot(axes, alpha=0, showedge=True)
            mesh.find_node(axes, showindex=True, color='r')
            mesh.find_node(axes, node=ipoints[cell2dof[0]], showindex=True, color='b')
            plt.show()

    def poisson_test(self, p=2):
        pde = CosCosCosData()
        mesh = self.plane_pmesh(n=4)
        space = PrismFiniteElementSpace(mesh, p=p)
        A = space.stiff_matrix()
        b = space.source_vector(pde.source)
        bc = BoundaryCondition(space, dirichlet=pde.dirichlet)
        uh = space.function()
        A, b = bc.apply_dirichlet_bc(A, b, uh)
        uh[:] = spsolve(A, b).reshape(-1)
        error = space.integralalg.L2_error(pde.solution, uh)
        print(error)

    def basis_test(self):
        pass

    def grad_basis_test(self):
        bcs, ws = self.space.integrator.get_quadrature_points_and_weights()
        self.space.grad_basis(bcs)
        print(self.space.cell_to_dof())

test = PrismFiniteElementSpaceTest()
#test.test_cell_to_dof(plot=True)
#test.test_grad_basis()
#test.face_to_dof_test(p=3, plot=True)
test.poisson_test()


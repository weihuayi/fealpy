#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.pde.poisson_1d import CosData
from fealpy.pde.poisson_2d import CosCosData
from fealpy.pde.poisson_3d import CosCosCosData
from fealpy.mesh import TriangleMesh


class LagrangeFiniteElementSpaceTest:
    def __init__(self):
        pass

    def test_space_on_triangle(self, p=3):
        pde = CosCosData()
        mesh = pde.init_mesh(0)
        space = LagrangeFiniteElementSpace(mesh, p=p)
        ips = space.interpolation_points()
        face2dof = space.dof.face_to_dof()
        print(face2dof)
        print(mesh.entity('cell'))
        print('cell2dof:', space.cell_to_dof())

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, node=ips, showindex=True)
        mesh.find_edge(axes, showindex=True)
        plt.show()

    def test_space_on_tet(self, p=3):
        pde = CosCosCosData()
        mesh = pde.init_mesh(0)
        space = LagrangeFiniteElementSpace(mesh, p=p)
        ips = space.interpolation_points()
        face2dof = space.dof.face_to_dof()
        print(face2dof)
        print(mesh.entity('cell'))
        print('cell2dof:', space.cell_to_dof())

        fig = pl.figure()
        axes = a3.Axes3D(fig)
        mesh.add_plot(axes, showedge=True)
        mesh.find_node(axes, node=ips, showindex=True)
        mesh.find_face(axes, showindex=True)
        pl.show()

    def plot_basis(self, p=2, q=20, plot=True):
        node = np.array([
            (0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3)/2)], dtype=np.float)
        cell = np.array([(0, 1, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        space = LagrangeFiniteElementSpace(mesh, p=p)

        bc = space.multi_index_matrix[1](q)/q
        node = mesh.bc_to_point(bc) #( NQ, 1, 2)
        phi = space.basis(bc) # (NQ, 1, ldof)

        fig = plt.figure()
        axes = fig.gca(projection='3d')
        axes.plot_trisurf(
                node[..., 0].reshape(-1), 
                node[..., 1].reshape(-1), 
                phi[..., 0].reshape(-1),
                cmap='rainbow', lw=0.0, antialiased=True)
        axes.plot_trisurf(
                node[..., 0].reshape(-1), 
                node[..., 1].reshape(-1), 
                phi[..., 1].reshape(-1),
                cmap='rainbow', lw=0.0, antialiased=True)
        plt.show()
        

        


test = LagrangeFiniteElementSpaceTest()
#test.test_space_on_triangle()
#test.test_space_on_tet()
test.plot_basis()





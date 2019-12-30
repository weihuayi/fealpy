import sys
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.mesh import PrismMesh
from fealpy.functionspace import PrismFiniteElementSpace
from fealpy.pde.poisson_2d import CosCosData


class PrismFiniteElementSpaceTest():
    def __init__(self, p=2):
        self.mesh = self.plane_pmesh()
        self.space =  PrismFiniteElementSpace(self.mesh, p=p)

    def plane_pmesh(self):
        pde = CosCosData()
        mesh = pde.init_mesh(n=0)
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NN = mesh.number_of_nodes()
        pnode = np.zeros((2*NN, 3), dtype=mesh.ftype)
        pnode[:NN, 0:2] = node
        pnode[NN:, 0:2] = node
        pnode[NN:, 2] = 1
        pcell = np.r_['1', cell, cell + NN]

        pmesh = PrismMesh(pnode, pcell)
        return pmesh

    def test_basis(self):
        pass

    def test_grad_basis(self):
        bcs, ws = self.space.integrator.get_quadrature_points_and_weights()
        self.space.grad_basis(bcs)
        print(self.space.cell_to_dof())

p = 1
test = PrismFiniteElementSpaceTest(p=p)
test.test_grad_basis()

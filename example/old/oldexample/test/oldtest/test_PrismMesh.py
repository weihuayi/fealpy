import sys
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.geometry import Sphere
from fealpy.mesh import PrismMesh
from fealpy.pde.poisson_2d import CosCosData

class PrismMeshTest:
    def __init__(self, meshtype):
        if meshtype is 'one':
            self.mesh = self.one_pmesh()
        elif meshtype is 'plane':
            self.mesh = self.plane_pmesh()
        elif meshtype is 'sphere':
            self.mesh = self.sphere_pmesh()

    def one_pmesh(self):
        node = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0]], dtype=np.float)
        cell = np.array([[0, 1, 2, 3, 4,  5]], dtype=np.int)
        pmesh = PrismMesh(node, cell)
        return pmesh

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

    def sphere_pmesh(self, n=3):
        s0 = Sphere(radius=1.0)
        s1 = Sphere(radius=1.2)

        mesh = s0.init_mesh()

        mesh.uniform_refine(n, surface=s0)

        NN = mesh.number_of_nodes()
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        newNode, d = s1.project(node)
        pnode = np.r_['0', node, newNode]
        pcell = np.r_['1', cell, cell + NN]

        pmesh = PrismMesh(pnode, pcell)
        return pmesh


    def test_mesh(self, plot=False):
        mesh = self.mesh
        print(mesh.entity('node'))
        print(mesh.entity('edge'))
        print(mesh.entity('face'))
        print(mesh.entity('cell'))

        if plot is True:
            fig = plt.figure()
            axes = Axes3D(fig)
            mesh.add_plot(axes, alpha=0, showedge=True)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def test_jacobi_matrix(self):
        from fealpy.quadrature import PrismQuadrature
        qf = PrismQuadrature(2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        self.mesh.jacobi_matrix(bcs)

    def test_cell_volume(self):
        vol = self.mesh.cell_volume()
        print(vol)


t = PrismMeshTest('plane')
t.test_jacobi_matrix()
t.test_cell_volume()
t.test_mesh(plot=True)

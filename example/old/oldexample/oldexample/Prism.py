import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import vtk
from vtk.util import numpy_support as vns
from vtkplotter import *
from fealpy.mesh.implicit_surface import Sphere
from fealpy.mesh import PrismMesh

from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import CPPFEMDof3d, PrismFiniteElementSpace


def one_pmesh():
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

def plane_pmesh():
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

def sphere_pmesh(n=3):
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


pmesh = one_pmesh()
qf = pmesh.integrator(2)
bcs, ws = qf.get_quadrature_points_and_weights()

fig = plt.figure()
axes = Axes3D(fig)
pmesh.add_plot(axes, alpha=0, showedge=True)
pmesh.find_node(axes, node=p[:, 0, :], showindex=True)
plt.show()

# pmesh = sphere_pmesh(n=6)
# dof = CPPFEMDof3d(pmesh, p=5)
# print(dof.number_of_global_dofs())
# print(dof.dpoints.shape)
# print(dof.cell2dof)
# fig = plt.figure()
# axes = Axes3D(fig)
# pmesh.add_plot(axes, alpha=0,  threshold=lambda bc: bc[:, 0] < 0.5)
# plt.show()

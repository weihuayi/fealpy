import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import vtk
from vtk.util import numpy_support as vns
from vtkplotter import *
from fealpy.mesh.implicit_surface import Sphere
from fealpy.mesh import PrismMesh

from fealpy.pde.poisson_2d import CosCosData


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


def sphere_pmesh():
    s0 = Sphere(radius=1.0)
    s1 = Sphere(radius=1.2)

    mesh = s0.init_mesh()

    mesh.uniform_refine(3, surface=s0)

    NN = mesh.number_of_nodes()
    node = mesh.entity('node')
    cell = mesh.entity('cell')

    newNode, d = s1.project(node)
    pnode = np.r_['0', node, newNode]
    pcell = np.r_['1', cell, cell + NN]

    pmesh = PrismMesh(pnode, pcell)
    return pmesh


def pmeshactor(node, cell):

    points = vtk.vtkPoints()
    points.SetData(vns.numpy_to_vtk(node))

    NC = len(cell)
    cells = vtk.vtkCellArray()
    cells.SetCells(NC, vns.numpy_to_vtkIdTypeArray(cell))

    celltype = 15

    vmesh = vtk.vtkUnstructuredGrid()
    vmesh.SetPoints(points)
    vmesh.SetCells(celltype, cells)

    gf = vtk.vtkUnstructuredGridGeometryFilter()
    gf.SetInputData(vmesh)
    gf.Update()

    pd = gf.GetOutput()

    return Actor(gf.GetOutput)


pmesh = plane_pmesh()
face = pmesh.entity('face')
print(face)
bdface = pmesh.boundary_face()
isTFace = bdface[:, -1] == -1e9

print(bdface[~isTFace])

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
pmesh.add_plot(axes)
plt.show()

import numpy as np

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

import vtk
from vtk.util import numpy_support as vns
from vtkplotter import *
from fealpy.mesh.implicit_surface import Sphere




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

s0 = Sphere(radius=1.0)
s1 = Sphere(radius=1.2)

mesh = s0.init_mesh()

mesh.uniform_refine(3, surface=s0)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
node = mesh.entity('node')
cell = mesh.entity('cell')


newNode, d = s1.project(node)
pnode = np.r_['0', node, newNode]
pcell = np.r_['1', cell, cell + NN]

actor = pmeshactor(pnode, pcell)

vp = Plotter()
vp.show(actor)



#fig = plt.figure()
#axes = fig.add_subplot(111, projection='3d')
#mesh.add_plot(axes)
#plt.show()

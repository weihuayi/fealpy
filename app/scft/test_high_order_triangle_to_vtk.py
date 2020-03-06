
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import vtk.util.numpy_support as vnp

from fealpy.geometry import Sphere
from fealpy.functionspace import SurfaceLagrangeFiniteElementSpace

order = 2
surface = Sphere()
surface.radius = 3.65 
mesh = surface.init_mesh()

mesh.uniform_refine(n=2, surface=surface)
femspace = SurfaceLagrangeFiniteElementSpace(mesh, surface, p=order)
ldof = femspace.number_of_local_dofs()

mesh = femspace.mesh
node = mesh.node
NC = mesh.number_of_cells()
idx = np.array([0, 3, 5, 1, 4, 2], dtype=np.int64)
cell2dof = np.int64(mesh.space.cell_to_dof())

cell2dof = np.r_['1', ldof*np.ones((NC, 1), dtype=np.int), cell2dof[:, idx]]


points = vtk.vtkPoints()
points.SetData(vnp.numpy_to_vtk(node))
cells = vtk.vtkCellArray()
cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell2dof))

uGrid =vtk.vtkUnstructuredGrid() 
uGrid.SetPoints(points)
uGrid.SetCells(22, cells)

uGrid.GetPointData().AddArray(vnp.numpy_to_vtk(node[:, 0]))


writer = vtk.vtkUnstructuredGridWriter()
writer.SetFileName('test.vtk')
writer.SetInputData(uGrid)
writer.Write()
# Visualize
mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(uGrid)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
#actor.GetProperty().SetRepresentationToWireframe()
actor.GetProperty().SetRepresentation(4)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

renderer.AddActor(actor)
renderer.SetBackground(.2, .3, .4)

renderWindow.Render()
renderWindowInteractor.Start()

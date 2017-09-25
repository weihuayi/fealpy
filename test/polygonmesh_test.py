"""
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.PolygonMesh import PolygonMesh

#reader = vtk.vtkUnstructuredGridReader()
#reader.SetFileName(sys.argv[1])
#reader.Update()
#
#mesh = reader.GetOutput()
#wmesh = dsa.WrapDataObject(mesh)
#point = wmesh.GetPoints()
#cell = wmesh.GetCells()
#cellLocation = wmesh.GetCellLocations()
#cellType = wmesh.GetCellTypes()

point = np.array([
    (0, 0), (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1),
    (0, 2), (1, 2), (2, 2)], dtype=np.float)
cell = np.array([1, 4, 0, 3, 0, 4, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7], dtype=np.int)
cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

pmesh = PolygonMesh(point, cell, cellLocation)
area = pmesh.area()

d = pmesh.angle()
print(d)
print(pmesh.point)
print(pmesh.ds.cell)
print(pmesh.ds.edge)
print(pmesh.ds.edge2cell)
print(area)
print(pmesh.ds.cell_to_cell().toarray())
print(pmesh.ds.cell_to_point().toarray())
print(pmesh.ds.cell_to_edge().toarray())

fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
pmesh.find_point(axes, showindex=True)
pmesh.find_edge(axes, showindex=True)
pmesh.find_cell(axes, showindex=True)
plt.show()

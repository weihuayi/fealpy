import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
import matplotlib.pyplot as plt
import sys


reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("./GNBC_SLIP_STICK/all_gnbc/test_0000000000.vtu")
reader.Update()
'''
mesh = reader.GetOutput()
wmesh = dsa.WrapDataObject(mesh)
point = wmesh.GetPoints()
cell = wmesh.GetCells()
cellLocation = wmesh.GetCellLocations()
print(point)
print(cell)
print(cellLocation)
'''

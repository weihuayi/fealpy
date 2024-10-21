
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from .PolygonMesh import PolygonMesh

from tvtk.api import tvtk, write_data

import numpy as np

def load_vtk_mesh(fileName):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(fileName)
    reader.Update()
    mesh = reader.GetOutput()
    wmesh = dsa.WrapDataObject(mesh)
    point = wmesh.GetPoints()
    cell = wmesh.GetCells()
    cellLocation = wmesh.GetCellLocations()
    cellType = wmesh.GetCellTypes()
    pmesh = PolygonMesh(point[:,[0,1]], cell, cellLocation, cellType)
    return pmesh

def write_vtk_mesh(mesh, fileName):
    node = mesh.entity('node')
    GD = mesh.geo_dimension()
    if GD == 2:
        node = np.concatenate((node, np.zeros((node.shape[0], 1),
            dtype=mesh.ftype)), axis=1)
    ug = tvtk.UnstructuredGrid(points=node)

    if mesh.meshtype is 'hex':
        cell_type = tvtk.Hexahedron().cell_type
        cell = mesh.ds.cell
    elif mesh.meshtype is 'tri':
        cell_type = tvtk.Triangle().cell_type
        cell = mesh.ds.cell
        for key, value in mesh.cellData.items():
            i = ug.cell_data.add_array(value)
            ug.cell_data.get_array(i).name=key
        for key, value in mesh.pointData.items():
            i = ug.point_data.add_array(value)
            ug.point_data.get_array(i).name = key
    elif mesh.meshtype is 'polyhedron':
        cell_type = tvtk.Polygon().cell_type
        NF, faces = mesh.to_vtk()
        cell = tvtk.CellArray()
        cell.set_cells(NF, faces)
    elif mesh.meshtype is 'polygon':
       cell_type = tvtk.Polygon().cell_type
       NC, cells = mesh.to_vtk()
       cell = tvtk.CellArray()
       cell.set_cells(NC, cells)
    elif mesh.meshtype is 'tet':
        cell_type = tvtk.Tetra().cell_type
        cell = mesh.ds.cell
        for key, value in mesh.cellData.items():
            i = ug.cell_data.add_array(value)
            ug.cell_data.get_array(i).name = key
        for key, value in mesh.pointData.items():
            i = ug.point_data.add_array(value)
            ug.point_data.get_array(i).name = key
    ug.set_cells(cell_type, cell) 
    write_data(ug, fileName)


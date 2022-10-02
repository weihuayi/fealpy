
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

def add_data_file(vtkFile, cellData, pointData, fieldData=None):
    """

    Reference
    [1] https://github.com/pyscience-projects/pyevtk/blob/f4df80587e34a485c921cfc6e35d3cbbfb8b1c44/pyevtk/hl.py#L49
    """
    # Point data
    if pointData:
        keys = list(pointData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next(
            (key for key in keys if isinstance(pointData[key], np.ndarray)), None
        )
        vectors = next((key for key in keys if isinstance(pointData[key], tuple)), None)
        vtkFile.openData("Point", scalars=scalars, vectors=vectors)
        for key in keys:
            data = pointData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Point")

    # Cell data
    if cellData:
        keys = list(cellData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next(
            (key for key in keys if isinstance(cellData[key], np.ndarray)), None
        )
        vectors = next((key for key in keys if isinstance(cellData[key], tuple)), None)
        vtkFile.openData("Cell", scalars=scalars, vectors=vectors)
        for key in keys:
            data = cellData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Cell")

    # Field data
    # https://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files#XML_VTK_files
    if fieldData:
        keys = list(fieldData.keys())
        vtkFile.openData("Field")  # no attributes in FieldData
        for key in keys:
            data = fieldData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Field")

def append_data_to_file(vtkFile, cellData, pointData, fieldData=None):
    # Append data to binary section
    if pointData is not None:
        keys = list(pointData.keys())
        for key in keys:
            data = pointData[key]
            vtkFile.appendData(data)

    if cellData is not None:
        keys = list(cellData.keys())
        for key in keys:
            data = cellData[key]
            vtkFile.appendData(data)

    if fieldData is not None:
        keys = list(fieldData.keys())
        for key in keys:
            data = fieldData[key]
            vtkFile.appendData(data)

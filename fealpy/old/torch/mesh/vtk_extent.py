"""

Notes
-----
    FEALPy 的 VTK 的扩展模块

Authors
------
Huayi Wei, weihuayi@xtu.edu.cn
"""

import torch
import numpy as np
import vtk
import vtk.util.numpy_support as vnp

from .vtkCellTypes import *

def write_to_vtu(fname, node, NC, cellType, cell, nodedata=None, celldata=None):
    """

    Notes
    -----
    """
    points = vtk.vtkPoints()
    node = node.cpu().numpy()
    cell = cell.cpu().numpy()
    points.SetData(vnp.numpy_to_vtk(node))

    cells = vtk.vtkCellArray()
    cell = cell.astype(np.int64)
    cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))

    mesh =vtk.vtkUnstructuredGrid() 
    mesh.SetPoints(points)
    mesh.SetCells(cellType, cells)

    pdata = mesh.GetPointData()

    if nodedata is not None:
        for key, val in nodedata.items():
            val = val.cpu().numpy()
            if val is not None:
                if len(val.shape) == 2 and val.shape[1] == 2:
                    shape = (val.shape[0], 3)
                    val1 = np.zeros(shape, dtype=val.dtype)
                    val1[:, 0:2] = val
                else:
                    val1 = val

                if val1.dtype == np.bool_:
                    d = vnp.numpy_to_vtk(val1.astype(np.int_))
                else:
                    d = vnp.numpy_to_vtk(val1[:])
                d.SetName(key)
                pdata.AddArray(d)

    if celldata is not None:
        cdata = mesh.GetCellData()
        for key, val in celldata.items():
            val = val.cpu().numpy()
            if val is not None:
                if len(val.shape) == 2 and val.shape[1] == 2:
                    shape = (val.shape[0], 3)
                    val1 = np.zeros(shape, dtype=val.dtype)
                    val1[:, 0:2] = val
                else:
                    val1 = val

                if val1.dtype == np.bool_:
                    d = vnp.numpy_to_vtk(val1.astype(np.int_))
                else:
                    d = vnp.numpy_to_vtk(val1[:])

                d.SetName(key)
                cdata.AddArray(d)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fname)
    writer.SetDataMode(vtk.VTK_BINARY)
    writer.SetInputData(mesh)
    writer.Write()



"""

Notes
-----
    FEALPy 的 VTK 的扩展模块

Authors
------
Huayi Wei, weihuayi@xtu.edu.cn
"""

import numpy as np
import vtk
import vtk.util.numpy_support as vnp

from .vtkCellTypes import *

from .backup.core import multi_index_matrix2d

def vtk_cell_index(p, celltype):
    """

    Notes
    -----
        获取 vtk cell 的相对于 fealpy 网格 cell 的顶点编号规则，用于把 FEALPy 中
        的 cell 顶点编号顺序转化为 vtk 的编号顺序。
    """
    if celltype == VTK_LAGRANGE_CURVE: 
        return np.arange(p+1, dtype=np.int_)
    elif celltype == VTK_LAGRANGE_TRIANGLE:
        tri = vtk.vtkLagrangeTriangle()
        ldof = (p + 1)*(p + 2)//2
        tri.GetPointIds().SetNumberOfIds(ldof)
        tri.GetPoints().SetNumberOfPoints(ldof)
        tri.Initialize()
        multiIndex = np.zeros((ldof, 3), dtype=np.int_)
        for i in range(ldof):
            tri.GetPointIds().SetId(i, i)
            tri.ToBarycentricIndex(i, multiIndex[i])
        s = np.sum(multiIndex[:, 1:], axis=-1)
        index = s*(s+1)//2 + multiIndex[:, 2]
        return index
    elif celltype == VTK_LAGRANGE_TETRAHEDRON:
        tet = vtk.vtkLagrangeTetra()
        ldof = (p + 1)*(p + 2)*(p + 3)//6
        tet.GetPointIds().SetNumberOfIds(ldof)
        tet.GetPoints().SetNumberOfPoints(ldof)
        tet.Initialize()
        multiIndex = np.zeros((ldof, 4), dtype=np.int_)
        for i in range(ldof):
            tet.GetPointIds().SetId(i, i)
            tet.ToBarycentricIndex(i, multiIndex[i])
        s0 = np.sum(multiIndex[:, 1:], axis=-1)
        s1 = np.sum(multiIndex[:, 2:], axis=-1)
        index = s0*(s0+1)*(s0+2)//6 + s1*(s1+1)//2 + multiIndex[:, 3]
        return index
    elif celltype == VTK_LAGRANGE_QUADRILATERAL:
        quad = vtk.vtkLagrangeQuadrilateral()
        orders = (p, p)
        sizes = (p + 1, p + 1)
        index = np.zeros(sizes[0]*sizes[1], dtype=np.int_)
        for i, loc in enumerate(np.ndindex(sizes)):
            idx = quad.PointIndexFromIJK(loc[0], loc[1], orders)
            index[idx] = i
        return index
    elif celltype == VTK_LAGRANGE_HEXAHEDRON:
        hexa = vtk.vtkLagrangeHexahedron()
        orders = (p, p, p)
        sizes = (p + 1, p + 1, p+1)
        index = np.zeros(sizes[0]*sizes[1]*sizes[2], dtype=np.int_)
        for i, loc in enumerate(np.ndindex(sizes)):
            idx = hexa.PointIndexFromIJK(loc[0], loc[1], loc[2], orders)
            index[idx] = i
        return index
    elif celltype == VTK_LAGRANGE_WEDGE:
        wedge = vtk.vtkLagrangeWedge()
        orders = (p, p, p)
        size = (orders[0] + 1) * (orders[0] + 2) * (orders[2] + 1)// 2
        index = np.zeros(size, dtype=np.int_)
        multiIndex = multi_index_matrix2d(p)
        i = 0
        for i0, i1 in multiIndex[:, 1:]:
            for i2 in range(p+1):
                idx = wedge.PointIndexFromIJK(i0, i1, i2, orders)
#                print(i, ":", i0, i1, i2, idx)        
                index[idx] = i  
                i += 1
        return index


def write_to_vtu(fname, node, NC, cellType, cell, nodedata=None, celldata=None):
    """

    Notes
    -----
    """
    points = vtk.vtkPoints()
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

def write_polyhedron_mesh_to_vtu(fname, mesh, nodedata=None, celldata=None):
    """

    Notes 多面体网格 to VTK PolyhedronMesh
    -----
    """
    points = vtk.vtkPoints()
    points.SetData(vnp.numpy_to_vtk(node))

    NC = len(cell)
    cell = cell.astype(np.int64)
    face = face.astype(np.int64)
    cell2face = cellface.astype(np.int64)

    uGrid = vtkUnstructuredGrid()
    uGrid.SetPoints(points)
    for i in range(NC):
        FacesIdList = vtkIdList()
        FacesIdList.InsertNextId(4)
        Face = face[cell2face[i]]

        for f in Face:
            FacesIdList.InsertNextId(len(f))
            [FacesIdList.InsertNextId(j) for j in f]

        uGrid.InsertNextCell(VTK_POLYHEDRON, FacesIdList)

    pdata = mesh.GetPointData()
    if nodedata is not None:
        for key, val in nodedata.items():
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
    writer.SetInputData(uGrid)
    writer.Write()


def mesh_to_vtu_file(mesh, filename, celldata=None, nodedata=None, fielddata=None):
    from pyevtk.hl import unstructuredGridToVTK

    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    NV = mesh.ds.number_of_nodes_of_cells()
    node = mesh.entity('node')
    cell = mesh.entity('cell').reshape(-1)
    cell_type = np.empty(NC, dtype=cell.dtype) 
    cell_type[:] = mesh.vtk_cell_type() 
    offset = np.arange(start=NV, stop=NC*NV+1, step=NV, dtype=cell.dtype)
    GD = mesh.geo_dimension()

    x = np.ascontiguousarray(node[:, 0])
    y = np.ascontiguousarray(node[:, 1])
    if GD == 2:
        z = np.zeros_like(x)
    elif GD == 3:
        z = np.ascontiguousarray(node[:, 2])

    unstructuredGridToVTK(filename, x, y, z, cell, offset, cell_type,
            cellData=celldata, pointData=nodedata, fieldData=fielddata)

    return filename 

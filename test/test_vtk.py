#!/usr/bin/env python

import vtk
import vtk.util.numpy_support as vnp
import time

import numpy as np
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import VTK_POLYHEDRON, vtkUnstructuredGrid
from fealpy.mesh import TetrahedronMesh
from fealpy.mesh.vtk_extent import write_polyhedron_mesh_to_vtu

def MakePolyhedron():
    """
      Make a regular dodecahedron. It consists of twelve regular pentagonal
      faces with three faces meeting at each vertex.
    """
    # numberOfVertices = 20
    numberOfFaces = 12
    # numberOfFaceVertices = 5

    points = vtkPoints()
    points.InsertNextPoint(1.21412, 0, 1.58931)
    points.InsertNextPoint(0.375185, 1.1547, 1.58931)
    points.InsertNextPoint(-0.982247, 0.713644, 1.58931)
    points.InsertNextPoint(-0.982247, -0.713644, 1.58931)
    points.InsertNextPoint(0.375185, -1.1547, 1.58931)
    points.InsertNextPoint(1.96449, 0, 0.375185)
    points.InsertNextPoint(0.607062, 1.86835, 0.375185)
    points.InsertNextPoint(-1.58931, 1.1547, 0.375185)
    points.InsertNextPoint(-1.58931, -1.1547, 0.375185)
    points.InsertNextPoint(0.607062, -1.86835, 0.375185)
    points.InsertNextPoint(1.58931, 1.1547, -0.375185)
    points.InsertNextPoint(-0.607062, 1.86835, -0.375185)
    points.InsertNextPoint(-1.96449, 0, -0.375185)
    points.InsertNextPoint(-0.607062, -1.86835, -0.375185)
    points.InsertNextPoint(1.58931, -1.1547, -0.375185)
    points.InsertNextPoint(0.982247, 0.713644, -1.58931)
    points.InsertNextPoint(-0.375185, 1.1547, -1.58931)
    points.InsertNextPoint(-1.21412, 0, -1.58931)
    points.InsertNextPoint(-0.375185, -1.1547, -1.58931)
    points.InsertNextPoint(0.982247, -0.713644, -1.58931)

    points.InsertNextPoint(1.21412, 3, 1.58931)
    points.InsertNextPoint(0.375185, 4.1547, 1.58931)
    points.InsertNextPoint(-0.982247, 3.713644, 1.58931)
    points.InsertNextPoint(-0.982247, 2.213644, 1.58931)
    points.InsertNextPoint(0.375185, 1.5547, 1.58931)
    points.InsertNextPoint(1.96449, 3, 0.375185)
    points.InsertNextPoint(0.607062, 4.86835, 0.375185)
    points.InsertNextPoint(-1.58931, 4.1547, 0.375185)
    points.InsertNextPoint(-1.58931, 2.1547, 0.375185)
    points.InsertNextPoint(0.607062, 1.26835, 0.375185)
    points.InsertNextPoint(1.58931, 4.1547, -0.375185)
    points.InsertNextPoint(-0.607062, 4.86835, -0.375185)
    points.InsertNextPoint(-1.96449, 3, -0.375185)
    points.InsertNextPoint(-0.607062, 1.86835, -0.375185)
    points.InsertNextPoint(1.58931, 1.1547, -0.375185)
    points.InsertNextPoint(0.982247, 3.713644, -1.58931)
    points.InsertNextPoint(-0.375185, 4.1547, -1.58931)
    points.InsertNextPoint(-1.21412, 3, -1.58931)
    points.InsertNextPoint(-0.375185, 1.8547, -1.58931)
    points.InsertNextPoint(0.982247, 2.713644, -1.58931)

    # Dimensions are [numberOfFaces][numberOfFaceVertices]
    Face = [
        [0, 1, 2, 3, 4],
        [0, 5, 10, 6, 1],
        [1, 6, 11, 7, 2],
        [2, 7, 12, 8, 3],
        [3, 8, 13, 9, 4],
        [4, 9, 14, 5, 0],
        [15, 10, 5, 14, 19],
        [16, 11, 6, 10, 15],
        [17, 12, 7, 11, 16],
        [18, 13, 8, 12, 17],
        [19, 14, 9, 13, 18],
        [19, 18, 17, 16, 15]
    ]

    Face0 = [[Face[i][j]+20 for j in range(5)] for i in range(12)]

    FacesIdList = vtkIdList()
    # Number faces that make up the cell.
    FacesIdList.InsertNextId(numberOfFaces)
    for face in Face:
        # Number of points in the face == numberOfFaceVertices
        FacesIdList.InsertNextId(len(face))
        # Insert the pointIds for that face.
        [FacesIdList.InsertNextId(i) for i in face]

    FacesIdList0 = vtkIdList()
    # Number faces that make up the cell.
    FacesIdList0.InsertNextId(numberOfFaces)
    for face in Face0:
        # Number of points in the face == numberOfFaceVertices
        FacesIdList0.InsertNextId(len(face))
        # Insert the pointIds for that face.
        [FacesIdList0.InsertNextId(i) for i in face]

    uGrid = vtkUnstructuredGrid()
    #uGrid.InsertNextCell(VTK_POLYHEDRON, FacesIdList)
    uGrid.InsertNextCell(VTK_POLYHEDRON, FacesIdList0)
    uGrid.SetPoints(points)

    fname = 'mmm.vtu'

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(uGrid)
    writer.Write()

def MakeTetMesh():
    node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float_)
    cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int_)
    mesh = TetrahedronMesh(node, cell)
    NF = mesh.number_of_faces()
    NC = mesh.number_of_cells()

    face = mesh.entity('face')
    cell = mesh.entity('cell')
    cell2face = mesh.ds.cell_to_face()

    face = np.c_[3*np.ones(len(face)), face].astype(np.int64)
    #face = face.astype(np.int64)
    cell = np.c_[4*np.ones(len(cell)), cell].astype(np.int64)



    points = vtk.vtkPoints()
    points.SetData(vnp.numpy_to_vtk(node))

    faces = vnp.numpy_to_vtkIdTypeArray(face.flatten())
    facesLoc = vnp.numpy_to_vtkIdTypeArray(np.arange(0, (NF+1)*3, 3,dtype='int64'),deep=1)

    cells = vtk.vtkCellArray()
    cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell.flatten()))
    cellsLoc = vnp.numpy_to_vtkIdTypeArray(np.arange(0, NC*4, 4,dtype='int64'),deep=1)

    cellType = vnp.numpy_to_vtk(vtk.VTK_POLYHEDRON*np.ones(NC, dtype='uint8'),deep=1)

    mesh =vtk.vtkUnstructuredGrid() 
    mesh.SetPoints(points)
    mesh.SetCells(cellType, cellsLoc, cells, facesLoc, faces)

    #Face = vtk.vtkCellArray()
    #Face.SetNumberOfCells(NF)
    #Face.SetCells(NF, faces)

    #cellType = vnp.numpy_to_vtk(vtk.VTK_POLYGON*np.ones(NF, dtype='uint8'),deep=1)
    #mesh.SetCells(cellType, facesLoc, Face)

    fname = 'mmm.vtu'

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(mesh)
    writer.Write()

def MakeTetMesh1():
    node = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float_)
    cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int_)
    mesh = TetrahedronMesh(node, cell)
    mesh.uniform_refine(6)
    NF = mesh.number_of_faces()
    NC = mesh.number_of_cells()


    node = mesh.entity('node')
    face = mesh.entity('face')
    cell = mesh.entity('cell')
    cell2face = mesh.ds.cell_to_face()

    points = vtk.vtkPoints()
    points.SetData(vnp.numpy_to_vtk(node))

    s = time.time()
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

    fname = 'mmm.vtu'

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(uGrid)
    writer.Write()
    e = time.time()
    print('t = ', e-s)






MakeTetMesh1()




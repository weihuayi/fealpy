#!/usr/bin/env python3
# 

import sys

import numpy as np
import math 
import vtk


if sys.argv[1] == 'segment':
    order = int(sys.argv[2]) 
    curve = vtk.vtkLagrangeCurve()
    nPoints = order + 1

    curve.GetPointIds().SetNumberOfIds(nPoints)
    curve.GetPoints().SetNumberOfPoints(nPoints)
    curve.Initialize()
    mi = np.zeros((nPoints, 2), dtype=np.int_)

elif sys.argv[1] == 'tri':
    order = int(sys.argv[2]) 
    tri = vtk.vtkLagrangeTriangle()
    nPoints = (order + 1)*(order + 2)//2

    tri.GetPointIds().SetNumberOfIds(nPoints)
    tri.GetPoints().SetNumberOfPoints(nPoints)
    tri.Initialize()

    mi = np.zeros((nPoints, 3), dtype=np.int_)

    print(nPoints)
    for i in range(nPoints):
        tri.GetPointIds().SetId(i, i)
        tri.ToBarycentricIndex(i, mi[i])
    print(mi)
    s = np.sum(mi[:, 1:], axis=-1)
    idx = s*(s+1)//2 + mi[:, 2]
    print(idx)

elif sys.argv[1] == 'tet':
    order = int(sys.argv[2]) 
    tet = vtk.vtkLagrangeTetra()
    nPoints = (order + 1)*(order + 2)*(order + 3)//6

    tet.GetPointIds().SetNumberOfIds(nPoints)
    tet.GetPoints().SetNumberOfPoints(nPoints)
    tet.Initialize()

    mi = np.zeros((nPoints, 4), dtype=np.int_)

    print(nPoints)
    for i in range(nPoints):
        tet.GetPointIds().SetId(i, i)
        tet.ToBarycentricIndex(i, mi[i])
    print(mi)
    s0 = np.sum(mi[:, 1:], axis=-1)
    s1 = np.sum(mi[:, 2:], axis=-1)
    idx = s0*(s0+1)*(s0+2)//6 + s1*(s1+1)//2 + mi[:, 3]
    print(np.sort(idx))
elif sys.argv[1] == 'quad':
    order = int(sys.argv[2])
    orders = (order, order)
    sizes = (order + 1, order + 1)
    quad = vtk.vtkLagrangeQuadrilateral()
    nPoints = (order + 1)*(order + 1)
    print(np.ndindex(sizes))
    loc_to_cart = np.empty(nPoints, dtype='object')
    for loc in np.ndindex(sizes):
        idx = quad.PointIndexFromIJK(loc[0], loc[1], orders)
        print(loc, idx)
        cart = np.array([c / o for (c, o) in zip(loc, orders)])
        loc_to_cart[idx] = cart

    print(loc_to_cart)
 

if False:
# Let’s make a sixth-order tetrahedron 
    order = 6
# The number of points for a sixth-order tetrahedron is 
    nPoints = (order + 1) * (order + 2) * (order + 3) // 6;
     
# Create a tetrahedron and set its number of points. Internally, Lagrange cells
# compute their order according to the number of points they hold. 
    tet = vtk.vtkLagrangeTetra() 
    tet.GetPointIds().SetNumberOfIds(nPoints) 
    tet.GetPoints().SetNumberOfPoints(nPoints)
    tet.Initialize()
     
    point = [0.,0.,0.]
    barycentricIndex = [0, 0, 0, 0]
     
# For each point in the tetrahedron...
    for i in range(nPoints):
      # ...we set its id to be equal to its index in the internal point array. 
      tet.GetPointIds().SetId(i, i)
     
      # We compute the barycentric index of the point... 
      tet.ToBarycentricIndex(i, barycentricIndex)

      print(i, ": ", barycentricIndex)
     
      # ...and scale it to unity.
      for j in range(3):
          point[j] = float(barycentricIndex[j]) / order
     
      # A tetrahedron comprised of the above-defined points has straight
      # edges.
      tet.GetPoints().SetPoint(i, point[0], point[1], point[2])
     
# Add the tetrahedron to a cell array 
    tets = vtk.vtkCellArray() 
    tets.InsertNextCell(tet)
     
# Add the points and tetrahedron to an unstructured grid 
    uGrid =vtk.vtkUnstructuredGrid() 
    uGrid.SetPoints(tet.GetPoints())
    uGrid.InsertNextCell(tet.GetCellType(), tet.GetPointIds())
     
# Visualize
    mapper = vtk.vtkDataSetMapper() 
    mapper.SetInputData(uGrid)
     
    actor = vtk.vtkActor() 
    actor.SetMapper(mapper)
     
    renderer = vtk.vtkRenderer() 
    renderWindow = vtk.vtkRenderWindow() 
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
     
    renderer.AddActor(actor) 
    renderer.SetBackground(.2, .3, .4)
     
    renderWindow.Render() 
    renderWindowInteractor.Start()

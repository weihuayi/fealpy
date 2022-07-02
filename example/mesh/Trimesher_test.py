#!/usr/bin/env python3
#

import argparse
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriMesher as Mesher 

from fealpy.geometry import dcircle, drectangle
from fealpy.geometry import ddiff, huniform

import gmsh

parser = argparse.ArgumentParser(description=
        """
        三角形网格生成示例
        """)

parser.add_argument('--tool', 
        default='distmesh', type=str, 
        help='网格剖分使用的工具，默认distmesh,还有gmsh,meshpy')

args = parser.parse_args()
tool = args.tool
if tool == 'distmesh':
    h =0.005
    fd1 = lambda p: dcircle(p,[0.2,0.2],0.05)
    fd2 = lambda p: drectangle(p,[0.0,2.2,0.0,0.41])
    fd = lambda p: ddiff(fd2(p),fd1(p))

    def fh(p):
        h = 0.003 + 0.05*fd1(p)
        h[h>0.01] = 0.01
        return h

    bbox = [0,3,0,1]
    pfix = np.array([(0.0,0.0),(2.2,0.0),(2.2,0.41),(0.0,0.41)],dtype=np.float64)

    mesh = tri.distmesh(h,fd,fh,bbox,pfix)
if tool =='gmsh':
    gmsh.initialize()

    gmsh.model.add("test")
    lc = 0.1

    gmsh.model.geo.addPoint(0.0,0.0,0.0,lc,1)
    gmsh.model.geo.addPoint(2.2,0.0,0.0,lc,2)
    gmsh.model.geo.addPoint(2.2,0.41,0.0,lc,3)
    gmsh.model.geo.addPoint(0.0,0.41,0.0,lc,4)

    gmsh.model.geo.addLine(1,2,1)
    gmsh.model.geo.addLine(2,3,2)
    gmsh.model.geo.addLine(3,4,3)
    gmsh.model.geo.addLine(4,1,4)

    gmsh.model.geo.addPoint(0.15,0.2,0.0,lc,5)
    gmsh.model.geo.addPoint(0.25,0.2,0.0,lc,6)
    gmsh.model.geo.addPoint(0.2,0.2,0.0,lc,7)

    gmsh.model.geo.add_circle_arc(6,7,5,5)
    gmsh.model.geo.add_circle_arc(5,7,6,6)

    gmsh.model.geo.addCurveLoop([1,2,3,4],1)
    gmsh.model.geo.add_curve_loop([5,6],2)

    gmsh.model.geo.addPlaneSurface([1,2],1)
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.field.add("Distance",1)
    gmsh.model.mesh.field.setNumbers(1,"CurvesList",[5,6])
    gmsh.model.mesh.field.setNumber(1,"Sampling",1000)

    gmsh.model.mesh.field.add("Threshold",2)
    gmsh.model.mesh.field.setNumber(2,"InField",1)
    gmsh.model.mesh.field.setNumber(2,"SizeMin",0.0005)
    gmsh.model.mesh.field.setNumber(2,"SizeMax",0.015)
    gmsh.model.mesh.field.setNumber(2,"DistMin",0.005)
    gmsh.model.mesh.field.setNumber(2,"DistMax",0.01)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.Algorithm",7)
    gmsh.model.mesh.generate(2)

    mesh = tri.gmsh_to_TriangleMesh()
if tool == 'meshpy':
    from fealpy.mesh.MeshFactory import circle_interval_mesh
    import math
    points = np.array([[0.0, 0.0],[2.2,0.0],[2.2,0.41],[0.0,0.41]],dtype = np.float64)
    facets = np.array([[0,1],[1,2],[2,3],[3,0]],dtype = np.int_)

    p, f = circle_interval_mesh([0.2,0.2],0.05,0.01)
    
    points = np.append(points,p,axis=0)
    facets = np.append(facets,f+4,axis=0)
    h = 0.01
    mesh = tri.meshpy(points,facets,h,hole_points=[[0.2,0.2]])

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

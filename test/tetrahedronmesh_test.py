
import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

import sys

from fealpy.mesh.TetrahedronMesh import TetrahedronMesh

ax0 = a3.Axes3D(pl.figure())

point = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0,-1]], dtype=np.float)

cell = np.array([
    [0, 1, 2, 3],
    [0, 2, 1, 4]], dtype=np.int)

mesh = TetrahedronMesh(point, cell)
c,_ = mesh.circumcenter()
ax0.plot(c[:, 0], c[:, 1], c[:,2], 'ro',markersize=20)
mesh.add_plot(ax0, showboundary=False)
mesh.print()


ax1 = a3.Axes3D(pl.figure())

point = np.array([
    [-1,-1,-1],
    [ 1,-1,-1], 
    [ 1, 1,-1],
    [-1, 1,-1],
    [-1,-1, 1],
    [ 1,-1, 1], 
    [ 1, 1, 1],
    [-1, 1, 1]], dtype=np.float) 

cell = np.array([
    [0,1,2,6],
    [0,5,1,6],
    [0,4,5,6],
    [0,7,4,6],
    [0,3,7,6],
    [0,2,3,6]], dtype=np.int)

mesh = TetrahedronMesh(point, cell)
mesh.add_plot(ax1, showboundary=False)
mesh.find_point(ax1)
mesh.find_edge(ax1)
mesh.find_face(ax1)
mesh.find_cell(ax1)
mesh.print()
pl.show()


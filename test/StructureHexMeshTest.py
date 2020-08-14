import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

import sys

from fealpy.mesh import StructureHexMesh 

box = [0, 1, 0, 1, 0, 1]
n = 2

mesh = StructureHexMesh(box, n, n, n)
print(mesh.node)
print(mesh.ds.edge)
print(mesh.ds.cell)
print(mesh.ds.face)
print(mesh.ds.face2cell)
print(mesh.ds.cell2edge)

axes = a3.Axes3D(pl.figure())
mesh.add_plot(axes, alpha=0)
mesh.find_node(axes, showindex=True, markersize=30, color='r')
mesh.find_edge(axes, showindex=True, markersize=40, color='g')
mesh.find_face(axes, showindex=True, markersize=50, color='b')
mesh.find_cell(axes, showindex=True, markersize=60, color='k')
pl.show()

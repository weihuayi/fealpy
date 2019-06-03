#!/usr/bin/env python3
# 
import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

import sys

from fealpy.mesh import HexahedronMesh  

from fealpy.mesh.simple_mesh_generator import cubehexmesh 

def hex_mesh():
    point = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]], dtype=np.float)

    cell = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int)

    return HexahedronMesh(point, cell)

cube = [ 0, 1, 0, 1, 0, 1]
mesh = cubehexmesh(cube, nx = 2, ny = 2, nz = 2)

cell = mesh.entity('cell')
print(cell)

axes = a3.Axes3D(pl.figure())

cell2edge = mesh.ds.cell_to_edge().flatten()
cell2face = mesh.ds.cell_to_face().flatten()

print(mesh.number_of_nodes())
print(mesh.number_of_edges())
print(mesh.number_of_faces())
print(mesh.number_of_cells())

mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, markersize=300, color='r')
mesh.find_edge(axes, index=cell2edge, showindex=True, markersize=400, color='g')
mesh.find_face(axes, index=cell2face, showindex=True, markersize=500, color='b')
mesh.find_cell(axes, showindex=True, markersize=600, color='k')
mesh.print()
pl.show()

#!/usr/bin/env python3
# 

import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import IntervalMesh


if True:
    node = np.array([[0], [0.5], [1]], dtype=np.float64) # (NN, 1) array
    cell = np.array([[0, 1], [1, 2]], dtype=np.int_) # (NN, 2) array

if False:
    node = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]], dtype=np.float) # (NN, 2)
    cell = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int) # (NC, 2) array

mesh = IntervalMesh(node, cell)

GD = mesh.geo_dimension()
TD = mesh.top_dimension()

print('GD:', GD)
print('TD:', TD)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
print("NN:", NN)
print("NC:", NC)


node = mesh.entity('node')
cell = mesh.entity('cell')

length = mesh.entity_measure('cell') # (NC, )
bc = mesh.entity_barycenter('cell') # (NC, 1)

print('cell length:', length)
print('cell barycenter:', bc)


#isMarkedCell = np.zeros((NC, ), dtype=np.bool_)
#isMarkedCell[0] = True
#mesh.refine(isMarkedCell)

#fig = plt.figure()
#axes = fig.gca()
#mesh.add_plot(axes)
#mesh.find_node(axes, showindex=True)


#mesh.uniform_refine(2)
#cell = mesh.entity('cell')
#print('cell:\n', cell)


cell2node = mesh.ds.cell_to_node() # cell 
node2cell = mesh.ds.node_to_cell() 
print("node2cell:\n", node2cell)

mesh.uniform_refine(n=2)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=False, markersize=20)
#mesh.find_cell(axes, showindex=True)
plt.show()

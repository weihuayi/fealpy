#!/usr/bin/env python3
# 

import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh



node = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]], dtype=np.float) # (NN, 2)

cell = np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int) # (NC, 3)


mesh = TriangleMesh(node, cell)

NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
NC = mesh.number_of_cells()

node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')

ebc = mesh.entity_barycenter('edge')
cbc = mesh.entity_barycenter('cell')

area = mesh.entity_measure('cell')
eh = mesh.entity_measure('edge')

cell2node = mesh.ds.cell_to_node() # cell
cell2edge = mesh.ds.cell_to_edge() # (NC, 3) 
cell2cell = mesh.ds.cell_to_cell() # (NC, 3)

edge2cell = mesh.ds.edge_to_cell() # (NE, 4)
edge2node = mesh.ds.edge_to_node() # edge
edge2edge = mesh.ds.edge_to_edge() # sparse, (NE, NE)

node2cell = mesh.ds.node_to_cell() # sparse, (NN, NC)
node2edge = mesh.ds.node_to_edge() # sparse, (NN, NE)
node2node = mesh.ds.node_to_node() # sparse, (NN, NN)

isBdNode = mesh.ds.boundary_node_flag() # (NN, ), bool
isBdEdge = mesh.ds.boundary_edge_flag() # (NE, ), bool
isBdCell = mesh.ds.boundary_cell_flag() # (NC, ), bool

bdNodeIdx = mesh.ds.boundary_node_index() # 
bdEdgeIdx = mesh.ds.boundary_edge_index() # 
bdCellIdx = mesh.ds.boundary_cell_index() # 


mesh.uniform_refine(1)

mesh.print()


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()


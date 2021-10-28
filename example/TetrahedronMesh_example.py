#!/usr/bin/env python3
#X

import sys

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from fealpy.mesh import TetrahedronMesh

node = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0]], dtype=np.float) # (NN, 3)

cell = np.array([[1, 2, 0, 3], [2, 4, 3, 1]], dtype=np.int) # (NC, 3)

mesh = TetrahedronMesh(node, cell)

NN = mesh.number_of_nodes() # 节点 node 个数
NE = mesh.number_of_edges() # 边 edge 个数
NF = mesh.number_of_faces() # 面 face 个数
NC = mesh.number_of_cells() # 单元 cell 个数

node = mesh.entity('node') # 节点数组，形状为 (NN,3)，储存节点坐标
edge = mesh.entity('edge') # 边数组，形状为 (NE, 2), 储存每条边的两个节点的编号
face = mesh.entity('face') # 面数组，形状为 (NF, 3), 储存构成三角形的三个节点编号
cell = mesh.entity('cell') # 单元数组，形状为 (NC,4),储存构成四边形的四个节点编号

ebc = mesh.entity_barycenter('edge') # (NE,3)，储存各边的重心坐标
fbc = mesh.entity_barycenter('face') # (NF,3)，储存各面的重心坐标
cbc = mesh.entity_barycenter('cell') # (NC,3), 储存各单元的重心坐标

area = mesh.entity_measure('cell') # (NC, 1), 每个单元的面积
face = mesh.entity_measure('face') # (NF, 1), 每个面的面积
eh = mesh.entity_measure('edge') # (NE, 1), 每条边的长度

cell2cell = mesh.ds.cell_to_cell() # (NC, 4)
cell2face = mesh.ds.cell_to_face() # (NC, 4)
cell2edge = mesh.ds.cell_to_edge() # (NC, 6)
cell2node = mesh.ds.cell_to_node() # cell
print('cell2cell:\n', cell2cell)
print('cell2face:\n', cell2face)
print('cell2edge:\n', cell2edge)
print('cell2node:\n', cell2node)

face2cell = mesh.ds.face_to_cell() # (NF, 4)
face2face = mesh.ds.face_to_face() # (NF, NF)
face2edge = mesh.ds.face_to_edge() # (NF, 3)
face2node = mesh.ds.face_to_node() # face
print('face2cell:\n', face2cell)
print('face2face:\n', face2face)
print("face2edge:\n", face2edge)
print('face2node:\n', face2node)

edge2cell = mesh.ds.edge_to_cell() # (NE, NC)
edge2face = mesh.ds.edge_to_face() # (NE, NF)
edge2node = mesh.ds.edge_to_node() # edge
edge2edge = mesh.ds.edge_to_edge() # sparse, (NE, NE)
print('edge2cell:\n',edge2cell)
print('edge2face:\n',edge2face)
print("edge2edge:\n",edge2edge)
print('edge2node:\n',edge2face)


node2cell = mesh.ds.node_to_cell() # sparse, (NN, NC)
node2face = mesh.ds.node_to_face() # sparse, (NN, NF)
node2edge = mesh.ds.node_to_edge() # sparse, (NN, NE)
node2node = mesh.ds.node_to_node() # sparse, (NN, NN)

print('node2cell:\n',node2cell)
print('node2face:\n',node2face)
print('node2edge:\n',node2edge)
print("node2node:\n",node2node)


isBdNode = mesh.ds.boundary_node_flag() # (NN, ), bool
isBdEdge = mesh.ds.boundary_edge_flag() # (NE, ), bool
isBdFace = mesh.ds.boundary_face_flag() # (NC, ), bool
isBdCell = mesh.ds.boundary_cell_flag() # (NC, ), bool

bdNodeIdx = mesh.ds.boundary_node_index() # 
bdEdgeIdx = mesh.ds.boundary_edge_index() # 
bdFaceIdx = mesh.ds.boundary_face_index() # 
bdCellIdx = mesh.ds.boundary_cell_index() # 


#mesh.uniform_refine(1)

mesh.print()


fig = plt.figure()
axes = Axes3D(fig)
mesh.add_plot(axes, alpha=0, showedge=True)
mesh.find_node(axes,showindex=True,fontsize=40)
mesh.find_edge(axes, showindex=True,fontsize=40)
mesh.find_cell(axes, showindex=True,fontsize=40)
plt.show()


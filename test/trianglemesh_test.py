import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh.TriangleMesh import TriangleMesh, TriangleMeshDataStructure

node = np.array([
    (0, 0), 
    (1, 0), 
    (0.5, 0.865)], dtype=np.float)
cell = np.array([
    (0, 1, 2)], dtype=np.int)
a = np.array([0, 1, 2])
tmesh = TriangleMesh(node, cell)
tmesh.uniform_refine(1)
node = tmesh.node
cell = tmesh.entity('cell')
print(cell)
N = node.shape[0]
tmeshstrcture = TriangleMeshDataStructure(N, cell)
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True)
tmesh.find_cell(axes, index=a, color='r')
plt.show()

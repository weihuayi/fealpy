import numpy as np

import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Tritree import Tritree


class AdaptiveMarker():
    def __init__(self):
        self.theta = 0.2

    def refine_marker(self, tmesh):
        cell = tmesh.entity('cell')
        isLeafCell = tmesh.is_leaf_cell()
        flag = (np.sum(cell == 0, axis=1) == 1) & isLeafCell
        idx, = np.where(flag)
        return idx

    def coarsen_marker(self, qtmesh):
        pass


node = np.array([
    (0, 0.5), 
    (0.865, 0), 
    (1.73, 0.5),
    (0.865, 1)], dtype=np.float)
cell = np.array([
    (1, 2, 3),
    (3, 0, 1)], dtype=np.int)

tmesh = TriangleMesh(node, cell)
tmesh.uniform_refine(0)

node = tmesh.entity('node')
cell = tmesh.entity('cell')
tritree = Tritree(node, cell, irule=1)
marker = AdaptiveMarker()

for i in range(1):
    tritree.refine(marker)



a = np.array([2, 4], dtype=np.int)
fig = plt.figure()
axes = fig.gca()
tritree.add_plot(axes)
#tritree.find_node(axes, showindex=True)
tritree.find_cell(axes, index=a)
#tritree.find_edge(axes, showindex=True) 
plt.show()





























#fig = plt.figure()
#axes = fig.gca()
#tmesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True)
#tmesh.find_cell(axes, showindex=True)
#plt.show()

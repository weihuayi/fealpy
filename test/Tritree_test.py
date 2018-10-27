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
        flag = (np.sum(cell == 5, axis=1) == 1) & isLeafCell
        idx, = np.where(flag)
        return idx

    def coarsen_marker(self, qtmesh):
        pass


node = np.array([
    (0, 0), 
    (1, 0), 
    (1, 1),
    (0, 1)], dtype=np.float)
cell = np.array([
    (1, 2, 0), 
    (3, 0, 2)], dtype=np.int)
tmesh = TriangleMesh(node, cell)
tmesh.uniform_refine(1)

node = tmesh.entity('node')
cell = tmesh.entity('cell')
tritree = Tritree(node, cell)
marker = AdaptiveMarker()

for i in range(2):
    tritree.refineRG(marker)




fig = plt.figure()
axes = fig.gca()
tritree.add_plot(axes)
tritree.find_node(axes, showindex=True)
tritree.find_cell(axes, showindex=True)
tritree.find_edge(axes, showindex=True) 
plt.show()





























#fig = plt.figure()
#axes = fig.gca()
#tmesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True)
#tmesh.find_cell(axes, showindex=True)
#plt.show()

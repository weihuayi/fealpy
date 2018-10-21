import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh.Tritree import Tritree

class AdaptiveMarker():
    def __init__(self, phi):
        self.phi = phi 

    def refine_marker(self, tmesh):
        node = tmesh.entity('node')
        cell = tmesh.entity('cell')
        idx = tmesh.leaf_cell_index()
        value = self.phi(node)
        valueSign = np.sign(value)
        valueSign[np.abs(value)<1e-12] = 0
        flag = (np.abs(np.sum(valueSign[cell[idx, :]], axis=1)) < 3)
        idx = idx[flag]
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

cxy = (0.5, 0.5)
r = 0.3
phi = lambda p: dcircle(p, cxy, r)
marker = AdaptiveMarker(phi)

circle = Circle(cxy, r, edgecolor='g', fill=False, linewidth=2)

mesh = TriangleMesh(node, cell)
mesh.uniform_refine()

node = mesh.entity('node')
cell = mesh.entity('cell')
tmesh = Tritree(node, cell)

for i in range(2):
    tmesh.refine(marker)

idx = tmesh.leaf_cell_index()
mesh = TriangleMesh(tmesh.node, tmesh.ds.cell[idx, :])
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
#mesh.find_cell(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True) 
plt.show()





























#fig = plt.figure()
#axes = fig.gca()
#tmesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True)
#tmesh.find_cell(axes, showindex=True)
#plt.show()

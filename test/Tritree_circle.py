import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh.tree_data_structure import Tritree
m = int(sys.argv[1])
class AdaptiveMarker():
    def __init__(self, phi):
        self.phi = phi 

    def refine_marker(self, tmesh):
        node = tmesh.entity('node')
        cell = tmesh.entity('cell')
        idx = tmesh.leaf_cell_index()
        value = self.phi(node)
        valueSign = np.sign(value)
        valueSign[np.abs(value)<1e-6] = 0
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
mesh.uniform_refine(1)

node = mesh.entity('node')
cell = mesh.entity('cell')
tmesh = Tritree(node, cell, irule=1)

for i in range(m):
    tmesh.refine(marker)


fig0 = plt.figure()
axes0 = fig0.gca()
tmesh.add_plot(axes0)
#tmesh.find_node(axes, showindex=True)
tmesh.find_cell(axes0, showindex=True)
#tmesh.find_edge(axes, showindex=True) 
pmesh = tmesh.to_conformmesh()
print(set(pmesh.celldata['idxmap']))
fig1 = plt.figure()
axes1 = fig1.gca()
pmesh.add_plot(axes1)
#pmesh.find_node(axes1, showindex=True)
pmesh.find_cell(axes1, showindex=True)
#pmesh.find_edge(axes1, showindex=True)
plt.show()





























#fig = plt.figure()
#axes = fig.gca()
#tmesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True)
#tmesh.find_cell(axes, showindex=True)
#plt.show()

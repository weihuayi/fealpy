import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh.Tritree import Tritree

def get_idx(tritree, phi):
    node = tritree.entity('node')
    cell = tritree.entity('cell')
    isLeafCell = tritree.is_leaf_cell()
    cell2node = tritree.ds.cell_to_node()
    value = phi(node)
    valueSign = np.sign(value)
    valueSign[np.abs(value)<1e-8] = 0
    return idx

node = np.array([
    (0, 0), 
    (1, 0), 
    (1, 1),
    (0, 1)], dtype=np.float)
cell = np.array([
    (1, 2, 0), 
    (3, 0, 2)], dtype=np.int)

cxy = (0.5, 0.5)
r = 0.5
phi = lambda p: dcircle(p, cxy, r)
circle = Circle(cxy, r, edgecolor='g', fill=False, linewidth=2)

tmesh = TriangleMesh(node, cell)
tmesh.uniform_refine()

node = tmesh.entity('node')
cell = tmesh.entity('cell')
tritree = Tritree(node, cell)

for i in range(5):
    idx = get_idx(tritree, phi);
    tritree.refine(idx);

fig = plt.figure()
axes = fig.gca()
tritree.add_plot(axes)
#tritree.find_node(axes, showindex=True)
#tritree.find_cell(axes, showindex=True)
#tritree.find_edge(axes, showindex=True) 
plt.show()





























#fig = plt.figure()
#axes = fig.gca()
#tmesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True)
#tmesh.find_cell(axes, showindex=True)
#plt.show()

import numpy as np

import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Tritree import Tritree

#def get_idx(tritree):
#    cell = tritree.entity('cell')
#    isLeafCell = tritree.is_leaf_cell()
#    flag = (np.sum(cell==5, axis=1) == 1) & isLeafCell
#    idx, = np.where(flag)
#    return idx
idx = np.arange(2, 7)
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

for i in range(1):
#    idx = get_idx(tritree); 
    tritree.refine(idx);
print(tritree.leaf_cell_index())
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

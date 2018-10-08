import numpy as np

import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Tritree import Tritree

node = np.array([
    (0, 0), 
    (1, 0), 
    (1, 1),
    (0, 1)], dtype=np.float)
cell = np.array([
    (1, 2, 0), 
    (3, 0, 2)], dtype=np.int)
tmesh = TriangleMesh(node, cell)
tmesh.uniform_refine(0)
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_node(axes, showindex=True)
tmesh.find_edge(axes, showindex=True) 
tmesh.find_cell(axes, showindex=True) 
plt.show()
tmesh = Tritree(tmesh.node, tmesh.ds.cell)
#print(tmesh.leaf_cell_index())
#print(tmesh.leaf_cell())
#print(tmesh.is_leaf_cell())
#print(tmesh.is_root_cell())
print(tmesh.refine())





























#fig = plt.figure()
#axes = fig.gca()
#tmesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True)
#tmesh.find_cell(axes, showindex=True)
#plt.show()

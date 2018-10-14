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
tmesh.uniform_refine()

node = tmesh.entity('node')
cell = tmesh.entity('cell')
tritree = Tritree(node, cell)
idx = np.arange(2, 7); 
ef =tritree.refine(idx);

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

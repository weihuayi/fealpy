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
        [1, 2, 0],
        [3, 0, 2]], dtype=np.int)

idx = np.array([1, 3, 5, 7])
mesh = TriangleMesh(node, cell)
mesh.uniform_refine(4)
node = mesh.entity('node')
cell = mesh.entity('cell')
tmesh = Tritree(node, cell)
for i in range(3):
    tmesh.coarsen(idx)

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_node(axes, showindex=True)
tmesh.find_edge(axes, showindex=True)
tmesh.find_cell(axes, showindex=True)
plt.show()

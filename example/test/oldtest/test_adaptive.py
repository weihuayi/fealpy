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

mesh = TriangleMesh(node, cell)
mesh.uniform_refine(0)
node = mesh.entity('node')
cell = mesh.entity('cell')
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
#mesh.find_node(axes, showindex=True)
#mesh.find_edge(axes, showindex=True)
#mesh.find_cell(axes, showindex=True)
tmesh = Tritree(node, cell)

ismarkedcell = np.array([True, False])
tmesh.refine(ismarkedcell)

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)

pmesh = tmesh.to_conformmesh()
fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)

plt.show()

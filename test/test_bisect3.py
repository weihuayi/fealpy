
import numpy as np
from fealpy.mesh.TetrahedronMesh import TetrahedronMesh

ax1 = a3.Axes3D(pl.figure())

node = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1]], dtype=np.float)

cell = np.array([
    [0, 1, 2, 6],
    [0, 5, 1, 6],
    [0, 4, 5, 6],
    [0, 7, 4, 6],
    [0, 3, 7, 6],
    [0, 2, 3, 6]], dtype=np.int)

mesh = TetrahedronMesh(node, cell)
fig = plt.figure()
mesh.add_plot(ax1)
mesh.find_node(ax1)
mesh.find_edge(ax1)
mesh.find_face(ax1)
mesh.find_cell(ax1)
plt.show()

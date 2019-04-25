
import numpy as np
from fealpy.mesh.TetrahedronMesh import TetrahedronMesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
mesh.bisect()
mesh.bisect()
mesh.bisect()
mesh.bisect()
fig = plt.figure()
axes = fig.gca(projection='3d')
mesh.add_plot(axes, alpha=0, showedge=True)
mesh.find_node(axes)
mesh.find_edge(axes)
mesh.find_cell(axes)
plt.show()

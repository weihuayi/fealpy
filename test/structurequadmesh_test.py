
import numpy as np
import matplotlib.pyplot as plt
import sys

from fealpy.mesh.StructureQuadMesh import StructureQuadMesh 
box = [0, 1, 0, 1]
n = 4
qmesh = StructureQuadMesh(box, n, n)
qmesh.print()


NN = qmesh.number_of_nodes()
NE = qmesh.number_of_edges()
NC = qmesh.number_of_cells()

X = np.zeros(NE + NC, dtype=np.float)

I = np.arange(NC, dtype=np.int)
J = I

fig = plt.figure()
axes = fig.gca()
qmesh.add_plot(axes)
qmesh.find_node(axes, showindex=True)
qmesh.find_edge(axes, showindex=True)
qmesh.find_cell(axes, showindex=True)
plt.show()

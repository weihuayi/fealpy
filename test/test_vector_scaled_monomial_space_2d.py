import numpy as np

import matplotlib.pyplot as plt
from fealpy.functionspace.vector_vem_space import VectorScaledMonomialSpace2d
from fealpy.mesh import PolygonMesh

p = 1
point = np.array([(0.5, 0.5)], dtype=np.float)
point = np.array([(1, 1)], dtype=np.float)

node = np.array([ (0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float)
cell = np.array([0, 1, 2, 3], dtype=np.int)
cellLocation = np.array([0, 4], dtype=np.int)

pmesh = PolygonMesh(node, cell, cellLocation)

space = VectorScaledMonomialSpace2d(pmesh, p)

phi = space.basis(point)
print('phi', phi)
gphi = space.grad_basis(point)
print('gphi', gphi)

fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
pmesh.find_node(axes, showindex=True)
pmesh.find_edge(axes, showindex=True)
pmesh.find_cell(axes, showindex=True)
plt.show()

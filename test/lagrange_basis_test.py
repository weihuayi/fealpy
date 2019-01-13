
import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.mesh.TriangleMesh import TriangleMesh


degree = int(sys.argv[1])

point = np.array([
    [0,0],
    [1,0],
    [1,1],
    [0,1]], dtype=np.float)
cell = np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int)

mesh = TriangleMesh(point, cell)
V = LagrangeFiniteElementSpace(mesh, degree)

bc = np.array([(0.1, 0.2, 0.7), (0.3, 0.4, 0.3)])
for b in bc:
    print(V.basis(b))

#print(V.basis_einsum(bc))

ipoints = V.interpolation_points()
cell2dof = V.cell_to_dof()
fig, axes = plt.subplots(1, 3)
mesh.add_plot(axes[0])
mesh.find_node(axes[0], node=ipoints, showindex=True)

for ax in axes.reshape(-1)[1:]:
    ax.axis('tight')
    ax.axis('off')

axes[1].table(cellText=cell, rowLabels=['0:', '1:'], loc='center', fontsize=100)
axes[1].set_title('cell', y=0.7)
axes[2].table(cellText=cell2dof, rowLabels=['0:', '1:'], loc='center',
        fontsize=100)
axes[2].set_title('cell2dof', y=0.6)
plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
plt.show()

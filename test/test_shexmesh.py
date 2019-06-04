import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt


import mpl_toolkits.mplot3d as a3

from fealpy.mesh import StructureHexMesh
from fealpy.pde.poisson_3d import CosCosCosData

n = 16
pde = CosCosCosData()
mesh = StructureHexMesh([0, 1, 0, 1, 0, 1], 16, 16, 16)

A, h2 = mesh.laplace_operator()


node = mesh.entity('node')
F = h2*pde.source(node)

# 边界条件处理
NN = mesh.number_of_nodes()
u = np.zeros(NN, dtype=np.float)
isBdNode = mesh.ds.boundary_node_flag()
u[isBdNode] = pde.dirichlet(node[isBdNode])

F = F - A@u
F[isBdNode] = u[isBdNode]


bdIdx = np.zeros(NN, dtype=np.int)
bdIdx[isBdNode] = 1
Tbd = spdiags(bdIdx, 0, NN, NN)
T = spdiags(1-bdIdx, 0, NN, NN)
A = T@A@T + Tbd

uh = spsolve(A, F)

u = pde.solution(node)

print(np.max(np.abs( u - uh)))

'''
axes = a3.Axes3D(plt.figure())
mesh.add_plot(axes, showedge=True)
mesh.find_node(axes, showindex=True, markersize=300, color='r')
axes.set_axis_on()
plt.show()
'''

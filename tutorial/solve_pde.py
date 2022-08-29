
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags

from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import LagrangeFiniteElementSpace

p = 3
pde = CosCosData()

mesh = pde.init_mesh(n=1)

space = LagrangeFiniteElementSpace(mesh, p=p)
gdof = space.number_of_global_dofs()

A = space.stiff_matrix()
b = space.source_vector(pde.source)

"""
edge = mesh.entity('edge')
edge2cell = mesh.ds.edge_to_cell() # (NE, 4)

isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])

NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
edge2dof = np.zeros((NE, p+1), dtype=np.int_)
edge2dof[:,  0] = edge[:, 0]
edge2dof[:, -1] = edge[:, 1] 

edge2dof[:, 1:-1] =  np.arange(NN, NE*(p-1)+NN).reshape(-1, p-1)

isBdDof = np.zeros(gdof, dtype=np.bool_)
isBdDof[edge2dof[isBdEdge]] = True
"""

isBdDof = space.is_boundary_dof() # (gdof, )

uh = np.zeros(gdof, dtype=np.float64)

ipoints = space.interpolation_points()

uh[isBdDof] = pde.dirichlet(ipoints[isBdDof])

F = b - A@uh
F[isBdDof] = uh[isBdDof]

bdIdx = np.zeros(gdof, dtype=np.int_)
bdIdx[isBdDof] = 1
Tbd = spdiags(bdIdx, 0, gdof, gdof) 
T = spdiags(1-bdIdx, 0, gdof, gdof)
A = T@A@T + Tbd

uh[:] = spsolve(A, F)

#uh[~isBdDof] = spsolve(A[:, ~isBdDof][~isBdDof, :], F[~isBdDof])

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

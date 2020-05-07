#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat

import matplotlib.pyplot as plt
from fealpy.pde.poisson_2d import  CosCosData 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

from fealpy.graph import metis


n = int(sys.argv[1]) # 初始网格加密次数
m = int(sys.argv[2]) # 网格分割的块数

p=1
pde = CosCosData() 
mesh = pde.init_mesh(n=n)
space = LagrangeFiniteElementSpace(mesh, p=p)
bc = BoundaryCondition(space, dirichlet=pde.dirichlet)
uh = space.function()
A = space.stiff_matrix()
F = space.source_vector(pde.source)
A, F = bc.apply_dirichlet_bc(A, F, uh)


uh[:] = spsolve(A, F)
error = space.integralalg.L2_error(pde.solution, uh)

edgecuts, parts = metis.part_mesh(mesh, nparts=m, entity='cell', contig=True)

NC = mesh.number_of_cells()
NE = mesh.number_of_edges()
NN = mesh.number_of_nodes()

cell = mesh.entity('cell') # (NC, 3)
edge = mesh.entity('edge') # (NE, 2)
edge2cell = mesh.ds.edge_to_cell() # (NE, 4) 

nodeFlag = np.zeros(NN, dtype=np.int)

nodeFlag[cell[parts == 0]] = 0
nodeFlag[cell[parts == 1]] = 1
nodeFlag[edge[parts[edge2cell[:, 0]] != parts[edge2cell[:, 1]]]] = 2

A00 = A[nodeFlag == 0, :][:, nodeFlag == 0]
A11 = A[nodeFalg == 1, :][:, nodeFlag == 1]
A22 = A[nodeFalg == 2, :][:, nodeFlag == 2]

A02 = A[nodeFlag == 0, :][:, nodeFlag == 2]
A12 = A[nodeFlag == 1, :][:, nodeFlag == 2]

uh0 = uh[nodeFlag == 0]
uh1 = uh[nodeFlag == 1]
uh2 = uh[nodeFlag == 2]

F0 = F[nodeFlag == 0]
F1 = F[nodeFlag == 1]
F2 = F[nodeFlag == 2]

AA = bmat([[A00, None, A02], [None, A11, A12], [A02.T, A12.T, A22]])


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, color=nodeFlag)
#mesh.find_cell(axes, color=parts)
plt.show()

#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.linear_elasticity_model import LShapeDomainData2d 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

n = int(sys.argv[1])
p = int(sys.argv[2])
scale = float(sys.argv[3])

E=1e+5
nu=0.3
pde = LShapeDomainData2d(E=E, nu=nu) 

mu = pde.mu
lam = pde.lam

mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=p)
bc = BoundaryCondition(space, dirichlet=pde.dirichlet, neuman=pde.neuman)
uh = space.function(dim=2)
A = space.linear_elasticity_matrix(mu, lam)
F = space.source_vector(pde.source, dim=2)

bc.apply_neuman_bc(F, is_neuman_boundary=pde.is_neuman_boundary)
A, F = bc.apply_dirichlet_bc(A, F, uh, is_dirichlet_boundary=pde.is_dirichlet_boundary)

uh.T.flat[:] = spsolve(A, F)

error = space.integralalg.L2_error(pde.displacement, uh)

bc  = mesh.entity_barycenter('edge')
isBdEdge = mesh.ds.boundary_edge_flag()

isNEdge = isBdEdge & pde.is_neuman_boundary(bc)
isDEdge = isBdEdge & pde.is_dirichlet_boundary(bc)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

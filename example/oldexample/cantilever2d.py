#!/usr/bin/env python3
# 
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csr_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve, cg, LinearOperator, spilu

from fealpy.pde.linear_elasticity_model import  BeamData2d
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC, NeumannBC

from mumps import DMumpsContext

n = int(sys.argv[1])
p = int(sys.argv[2])

pde = BeamData2d(E = 2*10**6, nu = 0.3)
mu = pde.mu
lam = pde.lam
mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=p, q=4)
uh = space.function(dim=2)

A = space.linear_elasticity_matrix(lam, mu)
#A = space.recovery_linear_elasticity_matrix(lam, mu)
F = space.source_vector(pde.source, dim=2)

bc = NeumannBC(space, pde.neumann, threshold=pde.is_neumann_boundary)
F = bc.apply(F)

bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary) 
A, F = bc.apply(A, F, uh)

ctx = DMumpsContext()
ctx.set_silent()
if ctx.myid == 0:
    ctx.set_centralized_sparse(A)
    x = F.copy()
    ctx.set_rhs(x) # Modified in place
    ctx.run(job=6) # Analysis + Factorization + Solve
    ctx.destroy() # Cleanup

uh.T.flat[:] = x
# uh.T.flat[:] = spsolve(A, F) # (2, gdof ).flat

uI = space.interpolation(pde.displacement, dim=2)
e = uh - uI
#error = np.sqrt(np.mean(e**2))
error = sum(np.sqrt(np.sum(e**2, axis=-1)))/sum(np.sqrt(np.sum(uI**2, axis=-1)))

print('error:\n', error)

bc = mesh.entity_barycenter('edge')

isBdEdge = pde.is_dirichlet_boundary(bc)
isBdEdge = pde.is_neumann_boundary(bc)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_edge(axes, index=isBdEdge)

scale = 100
mesh.node += scale*uh
mesh.add_plot(plt)
plt.show()

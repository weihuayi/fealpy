#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csr_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve, cg, LinearOperator, spilu

from fealpy.pde.linear_elasticity_model import  PolyModel3d 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

n = int(sys.argv[1])

pde = PolyModel3d(lam=10000, mu=1.0)
mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=1, q=3)
uh = space.function(dim=3)

A = space.linear_elasticity_matrix(pde.lam, pde.mu)
F = space.source_vector(pde.source, dim=3)

bc = DirichletBC(space, pde.dirichlet) 
A, F = bc.apply(A, F, uh)

uh.T.flat[:], info = cg(A, F, tol=1e-10)

error = space.integralalg.L2_error(pde.displacement, uh)

print('error:\n', error)




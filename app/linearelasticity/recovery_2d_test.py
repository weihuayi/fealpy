#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt


from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.solver import LinearElasticityRLFEMFastSolver

n = int(sys.argv[1])
d = int(sys.argv[2])

if d == 2:
    from fealpy.pde.linear_elasticity_model import  HuangModel2d  as PDE
elif d == 3:
    from fealpy.pde.linear_elasticity_model import  PolyModel3d  as PDE

pde = PDE(lam=100000, mu=1.0)
mu = pde.mu
lam = pde.lam
mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=1)

uh = space.function(dim=d)
isBdDof = space.set_dirichlet_bc(uh, pde.dirichlet)


if False:
    P = space.stiff_matrix()
    M, G = space.recovery_linear_elasticity_matrix(mu, lam, format=None)
    F = space.source_vector(pde.source, dim=d)

    solver = LinearElasticityRLFEMFastSolver(mu, lam, M, G, P, isBdDof)
    uh = solver.solve(uh, F)
else:
    P = space.stiff_matrix()
    M, G = space.recovery_linear_elasticity_matrix(mu, lam, format=None)

    A = space.recovery_linear_elasticity_matrix(mu, lam, format='csr')
    F = space.source_vector(pde.source, dim=d)
    solver = LinearElasticityRLFEMFastSolver(mu, lam, M, G, P, isBdDof)

    bc = DirichletBC(space, pde.dirichlet) 
    A, F = bc.apply(A, F, uh)
    uh = solver.cg(A, F, uh)

error = space.integralalg.L2_error(pde.displacement, uh)
print(error)

#uh.T.flat[:] = spsolve(A, F)
#A = space.linear_elasticity_matrix(mu, lam)

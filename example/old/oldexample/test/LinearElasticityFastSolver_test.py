#!/usr/bin/env python3
# 
import sys

import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import spsolve


from fealpy.pde.linear_elasticity_model import  PolyModel3d
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import CrouzeixRaviartFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

from fealpy.solver.fast_solver import LinearElasticityLFEMFastSolver_1

n = int(sys.argv[1])
lam = float(sys.argv[2])
mu = float(sys.argv[3])
stype = sys.argv[4]

pde = PolyModel3d(lam=lam, mu=mu) 

mesh = pde.init_mesh(n=n)

NN = mesh.number_of_nodes()
print("NN:", 3*NN)

space = LagrangeFiniteElementSpace(mesh, p=1)

bc = DirichletBC(space, pde.dirichlet)

uh = space.function(dim=3)
A = space.linear_elasticity_matrix(pde.lam, pde.mu, q=1)
F = space.source_vector(pde.source, dim=3)
A, F = bc.apply(A, F, uh)

I = space.rigid_motion_matrix()
S = space.stiff_matrix(2*pde.mu)
S = bc.apply_on_matrix(S)

solver = LinearElasticityLFEMFastSolver_1(A, S, I, stype=stype, drop_tol=1e-6,
        fill_factor=40) 

solver.solve(uh, F)
error = space.integralalg.error(pde.displacement, uh)
print(error)


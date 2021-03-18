
#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat

from fealpy.pde.linear_elasticity_model import  PolyModel3d as PDE
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.solver.LinearElasticityRLFEMFastSolver import LinearElasticityRLFEMFastSolver as FastSovler

n = int(sys.argv[1])

pde = PDE(lam=10.0, mu=1.0)
mu = pde.mu
lam = pde.lam
mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=1, q=3)
M, G = space.recovery_linear_elasticity_matrix(mu, lam, format=None)
F = space.source_vector(pde.source, dim=3)

uh = space.function(dim=3)    
isBdDof = space.set_dirichlet_bc(uh, pde.dirichlet)   

solver = FastSovler(lam, mu, M, G, isBdDof)

solver.solve(uh, F, tol=1e-8)

uI = space.interpolation(pde.displacement, dim=3)

e = uh - uI
error = np.sqrt(np.mean(e**2))
print(error)


#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt


from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

p = int(sys.argv[1])
n = int(sys.argv[2])
d = int(sys.argv[3])

if d == 2:
    from fealpy.pde.linear_elasticity_model import  HuangModel2d  as PDE
elif d == 3:
    from fealpy.pde.linear_elasticity_model import  PolyModel3d  as PDE


pde = PDE(lam=100000, mu=1.0)
mu = pde.mu
lam = pde.lam
mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=p)
bc = BoundaryCondition(space, dirichlet=pde.dirichlet)
uh = space.function(dim=d)
#A = space.linear_elasticity_matrix(mu, lam)
A = space.recovery_linear_elasticity_matrix(mu, lam)
F = space.source_vector(pde.source, dim=d)
A, F = bc.apply_dirichlet_bc(A, F, uh)
uh.T.flat[:] = spsolve(A, F)
error = space.integralalg.L2_error(pde.displacement, uh)
print(error)

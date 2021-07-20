#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.linear_elasticity_model import  BoxDomainData3d 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

n = int(sys.argv[1])
p = int(sys.argv[2])
scale = float(sys.argv[3])

pde = BoxDomainData3d() 

mu = pde.mu
lam = pde.lam
mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=p)
bc = BoundaryCondition(space, dirichlet=pde.dirichlet)
uh = space.function(dim=3)
A = space.linear_elasticity_matrix(lam, mu)
F = space.source_vector(pde.source, dim=3)
A, F = bc.apply_dirichlet_bc(A, F, uh, is_dirichlet_boundary=pde.is_dirichlet_boundary)
uh.T.flat[:] = spsolve(A, F)
node = mesh.node.copy()
mesh.node = node + scale*uh

fig = plt.figure()
axes = Axes3D(fig) 
mesh.add_plot(axes )
plt.show()

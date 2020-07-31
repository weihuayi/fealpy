#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.linear_elasticity_model import  BoxDomainData3d 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

n = int(sys.argv[1])
p = int(sys.argv[2])
scale = float(sys.argv[3])

pde = BoxDomainData3d() 
mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=p)
bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
uh = space.function(dim=3)
A = space.linear_elasticity_matrix(pde.mu, pde.lam, q=1)
F = space.source_vector(pde.source, dim=3)
A, F = bc.apply(A, F, uh)
uh.T.flat[:] = spsolve(A, F)

# 原始网格
mesh.add_plot(plt)

# 变形网格
mesh.node += scale*uh
mesh.add_plot(plt)

plt.show()

#!/usr/bin/env python3
# 

import sys 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import CrouzeixRaviartFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 

from fealpy.tools.show import showmultirate

# solver
from scipy.sparse.linalg import spsolve
import pyamg

n = int(sys.argv[1])
maxit = int(sys.argv[2])
d = int(sys.argv[3])

if d == 2:
    from fealpy.pde.poisson_2d import CosCosData as PDE
elif d == 3:
    from fealpy.pde.poisson_3d import CosCosCosData as PDE

pde = PDE()
mesh = pde.init_mesh(n=n)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    space = CrouzeixRaviartFiniteElementSpace(mesh)
    NDof[i] = space.number_of_global_dofs()
    bc = DirichletBC(space, pde.dirichlet) 

    uh = space.function()
    A = space.stiff_matrix()

    F = space.source_vector(pde.source)

    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F).reshape(-1)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        mesh.uniform_refine()

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

plt.show()

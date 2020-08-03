#!/usr/bin/env python3
# 

import sys

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import pyamg

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import RobinBC 

from fealpy.tools.show import showmultirate


p = int(sys.argv[1])
n = int(sys.argv[2])
maxit = int(sys.argv[3])
d = int(sys.argv[4])

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
    space = LagrangeFiniteElementSpace(mesh, p=p)

    NDof[i] = space.number_of_global_dofs()

    uh = space.function()
    A = space.stiff_matrix()
    F = space.source_vector(pde.source)

    bc = RobinBC(space, pde.robin)
    A, F = bc.apply(A, F)

    #uh[:] = spsolve(A, F).reshape(-1)

    ml = pyamg.ruge_stuben_solver(A)  
    uh[:] = ml.solve(F, tol=1e-12, accel='cg').reshape(-1)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        mesh.uniform_refine()

if d == 2:
    fig = plt.figure()
    axes = fig.gca(projection='3d')
    uh.add_plot(axes, cmap='rainbow')
elif d == 3:
    print('The 3d function plot is not been implemented!')

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

plt.show()

#!/usr/bin/env python3
# 

import sys 
import numpy as np
from scipy.sparse.linalg import spsolve
import pyamg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_2d import CosCosData 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import RobinBC 

from fealpy.tools.show import showmultirate

p = int(sys.argv[1])
maxit = int(sys.argv[2])

pde = CosCosData()
mesh = pde.init_mesh(n=4)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    space = LagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    uI = space.interpolation(pde.solution)
    uh = space.function()
    A = space.stiff_matrix()
    F = space.source_vector(pde.source)

    bc = RobinBC(space, pde.robin) 
    bc.apply(A, F) # Here is the case for pure Robin bc
    uh[:] = spsolve(A, F) # we add a addtional dof

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        mesh.uniform_refine()



fig = plt.figure()
axes = fig.gca(projection='3d')
uh.add_plot(axes, cmap='rainbow')

fig = plt.figure()
axes = fig.gca(projection='3d')
uI.add_plot(axes, cmap='rainbow')

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

plt.show()

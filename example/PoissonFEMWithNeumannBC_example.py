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
from fealpy.boundarycondition import NeumannBC 

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
mesh = pde.init_mesh(n=3)

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

    bc = NeumannBC(space, pde.neumann) 
    A, F = bc.apply(F, A=A) # Here is the case for pure Neumann bc, we also need modify A
    # bc.apply(F) # Not pure Neumann bc case
    uh[:] = spsolve(A, F)[:-1] # we add a addtional dof

    # Here can not work 
    #ml = pyamg.ruge_stuben_solver(A)  
    #x = ml.solve(F, tol=1e-12, accel='cg').reshape(-1)
    #uh[:] = x[-1]

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

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


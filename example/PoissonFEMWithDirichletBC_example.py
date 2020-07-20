#!/usr/bin/env python3
# 

import sys 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_2d import CosCosData 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 

from fealpy.tools.show import showmultirate

# solver
from fealpy.solver import PETScSolver
from scipy.sparse.linalg import spsolve
import pyamg

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
    print("The {}-th computation:".format(i))

    space = LagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    bc = DirichletBC(space, pde.dirichlet) 

    uh = space.function()
    if d == 2:
        A = space.stiff_matrix()
    elif d == 3:
        A = space.parallel_stiff_matrix(q=p)

    F = space.source_vector(pde.source)

    A, F = bc.apply(A, F, uh)


    #ml = pyamg.ruge_stuben_solver(A)  
    #uh[:] = ml.solve(F, tol=1e-12, accel='cg').reshape(-1)

    if d==2:
        uh[:] = spsolve(A, F).reshape(-1)
    elif d==3:
        solver = PETScSolver()
        solver.solve(A, F, uh)

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

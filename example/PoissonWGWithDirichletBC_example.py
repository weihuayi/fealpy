#!/usr/bin/env python3
# 

import sys 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_2d import CosCosData as PDE
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import WeakGalerkinSpace2d
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

# solver
from scipy.sparse.linalg import spsolve

p = int(sys.argv[1])
n = int(sys.argv[2])
maxit = int(sys.argv[3])


pde = PDE()
box = pde.domain()

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    mesh = MF.boxmesh2d(box, nx=n, ny=n, meshtype='poly') 
    space = WeakGalerkinSpace2d(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    bc = DirichletBC(space, pde.dirichlet) 

    uh = space.function()
    A = space.stiff_matrix()
    A += space.stabilizer_matrix()
    F = space.source_vector(pde.source)
    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F).reshape(-1)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        n *= 2
        



fig = plt.figure()
axes = fig.gca(projection='3d')
uh.add_plot(axes, cmap='rainbow')

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

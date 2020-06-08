#!/usr/bin/env python3
# 

import sys 
import numpy as np
from scipy.sparse.linalg import spsolve
import pyamg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from fealpy.pde.poisson_2d import CosCosData 
#from fealpy.pde.poisson_1d import CosData 
from fealpy.pde.poisson_3d import CosCosCosData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

from fealpy.tools.show import showmultirate

p = int(sys.argv[1])
maxit = int(sys.argv[2])

pde = CosCosCosData()
# pde = CosData()
# mesh = pde.init_mesh(n=3, meshtype='tri')
mesh = pde.init_mesh(n=3)

fig = plt.figure()
axes = fig.gca(projection='3d')
mesh.add_plot(axes)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    space = LagrangeFiniteElementSpace(mesh, p=p)

    NDof[i] = space.number_of_global_dofs()
    bc = BoundaryCondition(space, dirichlet=pde.dirichlet)
    uh = space.function()

    A = space.stiff_matrix()
    F = space.source_vector(pde.source)

    A, F = bc.apply_dirichlet_bc(A, F, uh)

    #uh[:] = spsolve(A, F).reshape(-1)

    ml = pyamg.ruge_stuben_solver(A)  
    uh[:] = ml.solve(F, tol=1e-12, accel='cg').reshape((-1,))

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    mesh.uniform_refine()



#fig = plt.figure()
#axes = fig.gca(projection='3d')
#uh.add_plot(axes, cmap='rainbow')

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

plt.show()

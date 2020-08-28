#!/usr/bin/env python3
# 

import sys 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import cartesian, barycentric
from fealpy.pde.surface_poisson import SphereSinSinSinData  as PDE
from fealpy.functionspace import ParametricLagrangeFiniteElementSpace
from fealpy.tools.show import showmultirate, show_error_table

# solver
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat

p = int(sys.argv[1])
n = int(sys.argv[2])
maxit = int(sys.argv[3])

pde = PDE()
surface = pde.domain()
mesh = pde.init_mesh(meshtype='quad', p=p) # p 次的拉格朗日四边形网格

mesh.uniform_refine(n=n)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$',
             '$|| u - u_I||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_I||_{\Omega, 0}$',
             '$|| u_I - u_h ||_{\Omega, \infty}$'
             ]
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.float64)

m = 4

for i in range(maxit):
    print("The {}-th computation:".format(i))

    space = ParametricLagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    A = space.stiff_matrix(variables='u')
    C = space.integral_basis()

    F = space.source_vector(pde.source)
    A = bmat([[A, C.reshape(-1, 1)], [C, None]], format='csr')
    F = np.r_[F, 0]

    uh = space.function()
    x = spsolve(A, F).reshape(-1)
    uh[:] = x[:-1]

    uI = space.interpolation(pde.solution)
    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    errorMatrix[2, i] = space.integralalg.error(pde.solution, uI.value)
    errorMatrix[3, i] = space.integralalg.error(pde.gradient, uI.grad_value)
    errorMatrix[4, i] = np.max(np.abs(uI - uh))

    if i < maxit-1:
        mesh.uniform_refine()

mesh.nodedata['uh'] = uh
mesh.nodedata['uI'] = uI 

print(errorMatrix)
mesh.to_vtk(fname='surface_with_solution.vtu')
show_error_table(NDof, errorType, errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

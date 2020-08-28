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
maxit = int(sys.argv[2])

pde = PDE()
surface = pde.domain()
mesh = pde.init_mesh(meshtype='quad', p=p) # p 次的拉格朗日四边形网格

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

    uh = space.function()
    A = space.stiff_matrix()
    C = space.integral_basis()
    barf = space.integralalg.mesh_integral(pde.source)/np.sum(space.cellmeasure)

    @cartesian
    def f(p):
        return pde.source(p) - barf 

    F = space.source_vector(f)
    A = bmat([[A, C.reshape(-1, 1)], [C, None]], format='csr')
    F = np.r_[F, 0]

    x = spsolve(A, F).reshape(-1)
    uh[:] = x[:-1]

    baru = space.integralalg.mesh_integral(pde.solution)/np.sum(space.cellmeasure)
    uh += baru

    uI = space.interpolation(pde.solution)
    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    errorMatrix[2, i] = space.integralalg.error(pde.solution, uI.value)
    errorMatrix[3, i] = space.integralalg.error(pde.gradient, uI.grad_value)
    errorMatrix[4, i] = np.max(np.abs(uI - uh))
    print(errorMatrix)

    if i < maxit-1:
        mesh.uniform_refine()

mesh.nodedata['uh'] = uh
mesh.nodedata['uI'] = uI 

mesh.to_vtk(fname='surface_with_solution.vtu')
show_error_table(NDof, errorType, errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

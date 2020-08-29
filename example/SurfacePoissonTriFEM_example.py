#!/usr/bin/env python3
# 

import sys 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh.SurfaceTriangleMeshOptAlg import SurfaceTriangleMeshOptAlg 
from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh

from fealpy.decorator import cartesian, barycentric
from fealpy.pde.surface_poisson import SphereSinSinSinData  as PDE
from fealpy.functionspace import ParametricLagrangeFiniteElementSpace
from fealpy.tools.show import showmultirate, show_error_table

# solver
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat

def opt_mesh(mesh):
    p = mesh.p
    surface = mesh.surface
    NCN = mesh.number_of_corner_nodes()
    node = mesh.entity('node')[:NCN].copy()
    cell = mesh.entity('cell')[:, [0, -p-1, -1]]
    tmesh = TriangleMesh(node, cell)
    alg = SurfaceTriangleMeshOptAlg(surface, tmesh)
    alg.run(maxit=10)

    node = alg.mesh.entity('node')
    cell = alg.mesh.entity('cell')
    mesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
    return mesh



p = int(sys.argv[1])
n = int(sys.argv[2])
maxit = int(sys.argv[3])

pde = PDE()
surface = pde.domain()
mesh = pde.init_mesh(meshtype='tri', p=p) # p 次的拉格朗日四边形网格
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

    mesh = opt_mesh(mesh)
    space = ParametricLagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()


    A = space.stiff_matrix(variables='u')
    C = space.integral_basis()
    F = space.source_vector(pde.source)

    NN = mesh.number_of_corner_nodes()
    NC = mesh.number_of_cells()
    uI = space.interpolation(pde.solution)
    delta = (A@uI - F)

    A = bmat([[A, C.reshape(-1, 1)], [C, None]], format='csr')
    F = np.r_[F, 0]

    uh = space.function()
    x = spsolve(A, F).reshape(-1)
    uh[:] = x[:-1]

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    errorMatrix[2, i] = space.integralalg.error(pde.solution, uI.value)
    errorMatrix[3, i] = space.integralalg.error(pde.gradient, uI.grad_value)
    errorMatrix[4, i] = np.max(np.abs(uI - uh))

    mesh.nodedata['uh'] = uh
    mesh.nodedata['uI'] = uI 
    mesh.nodedata['delta'] = delta
    mesh.nodedata['error'] = uI - uh

    mesh.to_vtk(fname='surface_with_solution' + str(i)+'.vtu')

    if i < maxit-1:
        mesh.uniform_refine()


print(errorMatrix)
show_error_table(NDof, errorType, errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.pde.timeharmonic_2d import CosSinData
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d 
from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

p = int(sys.argv[1])
n = int(sys.argv[2])
maxit = int(sys.argv[3])

pde = CosSinData()
mesh = pde.init_mesh(n=n, meshtype='tri')

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla\\times u - \\nabla\\times u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    space = FirstKindNedelecFiniteElementSpace2d(mesh, p=p)

    lspace = LagrangeFiniteElementSpace(mesh, p=p+1)

    gdof = space.number_of_global_dofs()
    NDof[i] = gdof 

    uh = space.function()

    A = space.curl_matrix() - space.mass_matrix()
    F = space.source_vector(pde.source)

    isBdDof = space.boundary_dof()
    bdIdx = np.zeros(gdof, dtype=np.int)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    A = T@A@T + Tbd
    F[isBdDof] = 0 
    uh[:] = spsolve(A, F)

    ruh = lspace.function(dim=2) # (gdof, 2)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.curl, uh.curl_value)

    if i < maxit-1:
        mesh.uniform_refine()



show_error_table(NDof, errorType, errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

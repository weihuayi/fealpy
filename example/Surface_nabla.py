#!/usr/bin/env python3
# 

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.surface_poisson import SphereSinSinSinData  as PDE
from fealpy.mesh import LagrangeTriangleMesh
from fealpy.functionspace import IsoLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table

# solver
from fealpy.solver import PETScSolver
from scipy.sparse.linalg import spsolve
import pyamg

p = int(sys.argv[1])
n = int(sys.argv[2])
maxit = int(sys.argv[3])

pde = PDE()
surface = pde.domain()
mesh = pde.init_mesh(n=n)

errorType = ['$|| u - u_h||_{\Omega,0}$'
             ]
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    node = mesh.entity('node')
    cell = mesh.entity('cell')

    lmesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
    space = IsoLagrangeFiniteElementSpace(lmesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    qf = lmesh.integrator(q=p+3, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    print('bcs', bcs.shape)
    gphi = lmesh.grad_shape_function(bcs, p=p) #(NQ,NC,ldof,TD)
    print('gphi', gphi.shape)
    Jh = lmesh.jacobi_matrix(bcs)
    print('Jh', Jh.shape)
    bc = lmesh.bc_to_point(bcs)
    print('bc', bc.shape)
    val = np.sum(bc, axis =-1)
    print('val', val.shape)
    val1 = (Jh.T/val.T).T
    J2 = (Jh.T/(val**3).T).T
    val2 = ((bc**2).T*J2.T).T
    JJ = val1 -val2
    print('JJ',JJ.shape)
    Je = Jh -JJ
    errorMatrix[0,i] = np.max(abs(Je))


    if i < maxit-1:
        mesh.uniform_refine()
print(errorMatrix)
#showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

show_error_table(NDof, errorType, errorMatrix)
#plt.show()


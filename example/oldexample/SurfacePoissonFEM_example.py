#!/usr/bin/env python3
# 

import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import cartesian, barycentric
from fealpy.pde.surface_poisson import SphereSinSinSinData  as PDE
from fealpy.functionspace import ParametricLagrangeFESpace

from fealpy.tools.show import showmultirate, show_error_table

# solver
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat

import ipdb

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        曲面上的任意次等参有限元方法
        """)

parser.add_argument('--sdegree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--mdegree',
        default=1, type=int,
        help='网格的阶数, 默认为 1 次.')

parser.add_argument('--mtype',
        default='tri', type=str,
        help='网格类型， 默认三角形网格.')

parser.add_argument('--nrefine',
        default=4, type=int,
        help='初始网格加密的次数, 默认初始加密 4 次.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

sdegree = args.sdegree
mdegree = args.mdegree
mtype = args.mtype
nrefine = args.nrefine
maxit = args.maxit

#ipdb.set_trace()
pde = PDE()
surface = pde.domain()
mesh = pde.init_mesh(meshtype=mtype, p=mdegree) # p 次的拉格朗日三角形网格

mesh.uniform_refine(n=nrefine)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$',
             '$|| u - u_I||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_I||_{\Omega, 0}$',
             '$|| u_I - u_h ||_{\Omega, \infty}$'
             ]
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

m = 4

for i in range(maxit):
    print("The {}-th computation:".format(i))

    space = ParametricLagrangeFESpace(mesh, p=sdegree)
    NDof[i] = space.number_of_global_dofs()

    uI = space.interpolation(pde.solution)

    A = space.stiff_matrix()
    C = space.integral_basis()
    F = space.source_vector(pde.source)

    NN = mesh.number_of_corner_nodes()
    NC = mesh.number_of_cells()
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
    mesh.nodedata['error'] = np.abs(uI - uh)

    mesh.to_vtk(fname='surface_with_solution' + str(i)+'.vtu')

    if i < maxit-1:
        mesh.uniform_refine()


show_error_table(NDof, errorType, errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

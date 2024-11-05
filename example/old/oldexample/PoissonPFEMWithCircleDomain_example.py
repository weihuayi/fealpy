#!/usr/bin/env python3
# 

import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_2d import CircleSinSinData as PDE
from fealpy.mesh import LagrangeTriangleMesh
from fealpy.functionspace import ParametricLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table

from scipy.sparse.linalg import spsolve


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单位圆上的任意次等参有限元方法
        """)

parser.add_argument('--sdegree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--mdegree',
        default=1, type=int,
        help='网格的阶数, 默认为 1 次.')

parser.add_argument('--nrefine',
        default=4, type=int,
        help='初始网格加密的次数, 默认初始加密 4 次.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

sdegree = args.sdegree
mdegree = args.mdegree
nrefine = args.nrefine
maxit = args.maxit

pde = PDE()

mesh = pde.init_mesh(n=nrefine)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$',
             '$|| u - u_I||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_I||_{\Omega, 0}$',
             '$|| u_I - u_h ||_{\Omega, \infty}$'
             ]
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)


node = mesh.entity('node')
cell = mesh.entity('cell')
mesh = LagrangeTriangleMesh(node, cell, p=mdegree, boundary=pde.curve)

m = 4

for i in range(maxit):
    print("The {}-th computation:".format(i))

    space = ParametricLagrangeFiniteElementSpace(mesh, p=sdegree)
    NDof[i] = space.number_of_global_dofs()
    bc = DirichletBC(space, pde.dirichlet) 


    uI = space.interpolation(pde.solution)
    A = space.stiff_matrix()
    F = space.source_vector(pde.source)

    NN = mesh.number_of_corner_nodes()
    NC = mesh.number_of_cells()


    uh = space.function()
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    errorMatrix[2, i] = space.integralalg.error(pde.solution, uI.value)
    errorMatrix[3, i] = space.integralalg.error(pde.gradient, uI.grad_value)
    errorMatrix[4, i] = np.max(np.abs(uI - uh))

    mesh.nodedata['uh'] = uh
    mesh.nodedata['uI'] = uI 
    mesh.nodedata['error'] = np.abs(uI - uh)

    mesh.to_vtk(fname='circle_domain' + str(i)+'.vtu')

    if i < maxit-1:
        mesh.uniform_refine()


show_error_table(NDof, errorType, errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

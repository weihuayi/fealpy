#!/usr/bin/env python3
# 

import argparse

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

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        多边形网格上任意次 WG 有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help=' WG 空间的次数, 默认为 1 次.')

parser.add_argument('--ns',
        default=10, type=int,
        help='初始网格部分段数, 默认 10 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
ns = args.ns
maxit = args.maxit


pde = PDE()
box = pde.domain()

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.float64)

for i in range(maxit):
    mesh = MF.boxmesh2d(box, nx=ns, ny=ns, meshtype='poly') 
    space = WeakGalerkinSpace2d(mesh, p=degree)
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
        ns *= 2
        
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
uh.add_plot(axes, cmap='rainbow')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

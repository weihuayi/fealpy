#!/usr/bin/env python3
#

import argparse 
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        多边形网格上的任意次协调虚单元方法  
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='虚单元空间的次数, 默认为 1 次.')

parser.add_argument('--nx',
        default=4, type=int,
        help='x 方向剖分段数， 默认 4 段.')

parser.add_argument('--ny',
        default=4, type=int,
        help='y 方向剖分段数， 默认 4 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
nx = args.nx
ny = args.ny
maxit = args.maxit

pde = CosCosData()
domain = pde.domain()


errorType = ['$|| u - \Pi u_h||_{\Omega,0}$',
             '$||\\nabla u - \Pi \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.float64)

for i in range(maxit):
    mesh = MF.boxmesh2d(domain, nx=nx, ny=ny, meshtype='poly') 
    space = ConformingVirtualElementSpace2d(mesh, p=degree)
    NDof[i] = space.number_of_global_dofs()
    uh = space.function()
    bc = DirichletBC(space, pde.dirichlet)
    A = space.stiff_matrix()
    F = space.source_vector(pde.source)

    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F).reshape(-1)

    sh = space.project_to_smspace(uh)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, sh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, sh.grad_value)

    uI = space.interpolation(pde.solution)

    nx *= 2
    ny *= 2

mesh.add_plot(plt)
uh.add_plot(plt, cmap='rainbow')
showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=20, lw=2,
        ms=4)
plt.show()

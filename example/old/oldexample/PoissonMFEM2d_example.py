#!/usr/bin/env python3
#

"""
混合元求解 Poisson 方程, 

.. math::
    -\Delta u = f

转化为

.. math::
    (\mathbf u, \mathbf v) - (p, \\nabla\cdot \mathbf v) &= - <p, \mathbf v\cdot n>_{\Gamma_D}
           - (\\nabla\cdot\mathbf u, w) &= - (f, w), w \in L^2(\Omega)

"""

import argparse 

import matplotlib.pyplot as plt

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData as PDE
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.solver import SaddlePointFastSolver
from fealpy.tools.show import showmultirate


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        二维混合有限元方法求解 Poisson 方程

        """)

parser.add_argument('--degree',
        default=0, type=int,
        help='RT 有限元空间的次数, 默认为 0 次, 即 RT0 元.')

parser.add_argument('--nrefine',
        default=4, type=int,
        help='初始网格加密的次数, 默认初始加密 4 次.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
nrefine = args.nrefine
maxit = args.maxit


pde = PDE() 
mesh = pde.init_mesh(n=nrefine)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$|| p - p_h||_{\Omega,0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    space = RaviartThomasFiniteElementSpace2d(mesh, p=degree)

    udof = space.number_of_global_dofs()
    pdof = space.smspace.number_of_global_dofs()
    gdof = udof + pdof
    NDof[i] = gdof

    uh = space.function() # 速度空间
    ph = space.smspace.function() # 压力空间

    M = space.mass_matrix()
    B = -space.div_matrix()

    F0 = -space.set_neumann_bc(pde.dirichlet) # Poisson 的 D 氏边界变为 Neumann
    F1 = -space.smspace.source_vector(pde.source)

    if True:
        solver = SaddlePointFastSolver((M, B, None), (F0, F1))
        uh[:], ph[:] = solver.solve()
    else:
        AA = bmat([[M, B], [B.T, None]], format='csr')
        FF = np.r_['0', F0, F1]
        x = spsolve(AA, FF).reshape(-1)
        uh[:] = x[:udof]
        ph[:] = x[udof:]
    errorMatrix[0, i] = space.integralalg.error(pde.flux, uh.value, power=2)
    errorMatrix[1, i] = space.integralalg.error(pde.solution, ph.value, power=2) 

    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

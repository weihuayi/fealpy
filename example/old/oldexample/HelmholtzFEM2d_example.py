#!/usr/bin/env python3
# 

import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import RobinBC 

from fealpy.pde.helmholtz_2d import HelmholtzData2d
from fealpy.pde.helmholtz_3d import HelmholtzData3d

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形网格上任意次有限元方法求解二维 Helmholtz 方程 
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--wavenum', 
        default=1, type=int,
        help='模型的波数, 默认为 1.')

parser.add_argument('--cip', nargs=2,
        default=[0, 0], type=float,
        help=' CIP-FEM 的系数, 默认取值 0, 即标准有限元方法.')

parser.add_argument('--ns',
        default=20, type=int,
        help='初始网格 x 和 y 方向剖分段数, 默认 20 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()
degree = args.degree
k = args.wavenum
c = complex(args.cip[0], args.cip[1])
ns = args.ns
maxit = args.maxit

pde = HelmholtzData2d(k=k) 
domain = pde.domain()

errorType = ['$|| u - u_I||_{\Omega,0}$',
             '$|| \\nabla u - \\nabla u_I||_{\Omega, 0}$',
             '$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$',
             ]

errorMatrix = np.zeros((4, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):

    n = ns*(2**i)
    mesh = MF.boxmesh2d(domain, nx=n, ny=n, meshtype='tri')
    space = LagrangeFiniteElementSpace(mesh, p=degree)

    NDof[i] = space.number_of_global_dofs()

    uh = space.function(dtype=np.complex128)
    S = space.stiff_matrix()
    M = space.mass_matrix()
    P = space.penalty_matrix()
    F = space.source_vector(pde.source)
    A = S -  pde.k**2*M + c*P

    bc = RobinBC(space, pde.robin)
    A, F = bc.apply(A, F)

    uh[:] = spsolve(A, F)

    print("线性系统求解残量：", np.linalg.norm(np.abs(A@uh-F)))

    uI = space.interpolation(pde.solution)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uI)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uI.grad_value) 
    errorMatrix[2, i] = space.integralalg.error(pde.solution, uh)
    errorMatrix[3, i] = space.integralalg.error(pde.gradient, uh.grad_value) 


bc = np.array([1/3, 1/3, 1/3])
ps = mesh.bc_to_point(bc)
u = pde.solution(ps)
uI = uI(bc)
uh = uh(bc)


fig, axes = plt.subplots(2, 2)
mesh.add_plot(axes[0, 0], cellcolor=np.real(u), linewidths=0)
mesh.add_plot(axes[0, 1], cellcolor=np.imag(u), linewidths=0) 
mesh.add_plot(axes[1, 0], cellcolor=np.real(uh), linewidths=0)
mesh.add_plot(axes[1, 1], cellcolor=np.imag(uh), linewidths=0) 
plt.show()

if maxit > 1:
    showmultirate(plt, maxit-2, NDof, errorMatrix,  errorType, propsize=20)
    show_error_table(NDof, errorType, errorMatrix)

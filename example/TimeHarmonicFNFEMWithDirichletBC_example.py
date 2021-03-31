#!/usr/bin/env python3
# 
import sys
import argparse

import numpy as np
from numpy.linalg import inv
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.pde.timeharmonic_2d import CosSinData, LShapeRSinData
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d 
from fealpy.boundarycondition import DirichletBC 

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        这是一个自适应求解时谐方程的程序
        """)

parser.add_argument('--order',
        default=0, type=int,
        help='第一类 Nedlec 元的次数, 默认为 0!')

parser.add_argument('--size',
        default=8, type=int,
        help='初始网格的 x 和 y 方向剖分段数, 默认为 8 段')

parser.add_argument('--maxit',
        default=4, type=int,
        help='网格一致加密次数, 默认加密  4 次')

parser.print_help()
args = parser.parse_args()
print('程序参数为:', args)


pde = CosSinData()
mesh = pde.init_mesh()

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla\\times u - \\nabla\\times u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((len(errorType), args.maxit), dtype=np.float)
NDof = np.zeros(args.maxit, dtype=np.float)

for i in range(args.maxit):
    space = FirstKindNedelecFiniteElementSpace2d(mesh, p=args.order)
    bc = DirichletBC(space, pde.dirichlet)

    gdof = space.number_of_global_dofs()
    NDof[i] = gdof
    uh = space.function()
    A = space.curl_matrix() - space.mass_matrix()
    F = space.source_vector(pde.source)

    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F)
    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.curl, uh.curl_value)
    mesh.uniform_refine()

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

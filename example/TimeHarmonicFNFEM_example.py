#!/usr/bin/env python3
# 
import sys
import argparse

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.pde.timeharmonic_2d import CosSinData
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 

from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table


def curl_recover(uh):

    mesh = uh.space.mesh
    space = LagrangeFiniteElementSpace(mesh, p=1)
    ruh = space.function() # (gdof, 2)

    bc = np.array([1/3, 1/3, 1/3], dtype=mesh.ftype)
    val = uh.curl_value(bc) #(NC, )
    w = 1/mesh.entity_measure('cell')
    val *= w

    NN = mesh.number_of_nodes() 
    NC = mesh.number_of_cells()
    cell = mesh.entity('cell')
    w = np.broadcast_to(w.reshape(-1, 1), shape=cell.shape)
    W = np.zeros(NN, dtype=mesh.ftype)
    np.add.at(W, cell, w)

    val = np.broadcast_to(val.reshape(-1, 1), shape=cell.shape)
    np.add.at(ruh, cell, val)
    ruh /= W

    return ruh

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        这是一个自适应求解时谐方程的程序
        """)

parser.add_argument('--order', 
        default=0, type=int,
        help='第一类 Nedlec 元的次数, 默认为 0, 注意目前高次的还没有测试成功!')

parser.add_argument('--size', 
        default=5, type=int,
        help='初始网格的 x 和 y 方向剖分段数, 默认为 5 次')

parser.add_argument('--maxit', 
        default=40, type=int,
        help='自适应迭代次数, 默认自适应迭代 30 次')

parser.add_argument('--theta', 
        default=0.3, type=float,
        help='自适应迭代的 theta 参数, 默认为  0.3')

parser.print_help()
args = parser.parse_args()
print('程序参数为:', args)


## 开始计算

pde = CosSinData()
mesh = MeshFactory.boxmesh2d(pde.domain(), nx=args.size, ny=args.size, meshtype='tri') 

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla\\times u - \\nabla\\times u_h||_{\Omega, 0}$',
             '$||\\nabla\\times u - G(\\nabla\\times u_h)||_{\Omega, 0}$',
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

    ruh = curl_recover(uh)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.curl, uh.curl_value)
    errorMatrix[2, i] = space.integralalg.L2_error(pde.curl, ruh)
    eta = space.integralalg.error(uh.curl_value, ruh, power=2, celltype=True) # 计算单元上的恢复型误差

    if i < args.maxit - 1:
        isMarkedCell = mark(eta, theta=args.theta)
        mesh.bisect(isMarkedCell)
        mesh.add_plot(plt)
        plt.savefig('./test-' + str(i+1) + '.png')
        plt.close()


showmultirate(plt, args.maxit-10, NDof, errorMatrix,  errorType, propsize=20)
plt.show()

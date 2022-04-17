#!/usr/bin/env python3
# 

import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_2d import CosCosData as PDE
from fealpy.mesh import StructureQuadMesh 
from fealpy.tools.show import showmultirate

from scipy.sparse.linalg import spsolve


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        在二维笛卡尔网格上求解定义在矩形区域上带 Dirichlet 边界的 Poisson 方程 
        """)

parser.add_argument('--nx',
        default=10, type=int,
        help='初始笛卡尔网格 x 方向剖分段数, 默认 10 段.')

parser.add_argument('--ny',
        default=10, type=int,
        help='初始笛卡尔网格 y 方向剖分段数, 默认 10 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

nx = args.nx
ny = args.ny
maxit = args.maxit


pde = PDE()
box = pde.domain()

errorType = ['$|| u - u_h||_{\in,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)
hs = np.zeros(maxit, dtype=np.float64)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    # 0. 创建结构网格对象
    mesh = StructureQuadMesh(box, nx=nx, ny=ny)
    hs[i] = max(mesh.hx, mesh.hy)

    # 1. 组装矩阵和右端项 
    A = mesh.laplace_operator()
    F = mesh.interpolation(pde.source)

    # 2. 创建解向量数组
    NN = mesh.number_of_nodes()
    NDof[i] = NN
    uh = np.zeros(NN, dtype=np.float64)

    # 3. 处理 Dirichlet 边界条件
    node = mesh.entity('node')
    isBdNode = mesh.ds.boundary_node_flag() # 所有边界点都设为 Drichlet 点
    uh[isBdNode] = pde.dirichlet(node[isBdNode])

    isInNode = ~isBdNode
    F -= A@uh

    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isInNode] = 1
    Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    A = T@A@T + Tbd

    F[isBdNode] = uh[isBdNode]

    # 4. 计算误差
    uI = mesh.interpolation(pde.solution)

    if i < maxit-1:
        mesh.uniform_refine()

fig = plt.figure()
axes = fig.add_subplot(1, 2, 1, projection='3d')
uh.add_plot(axes, cmap='rainbow')

axes = fig.add_subplot(1, 2, 2)
showmultirate(axes, 0, NDof, errorMatrix,  errorType, 
        propsize=40)

plt.show()

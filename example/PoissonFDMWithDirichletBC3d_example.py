#!/usr/bin/env python3
# 

import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_3d import CosCosCosData as PDE
from fealpy.mesh import StructureHexMesh 
from fealpy.tools.show import showmultirate

from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        在三维笛卡尔网格上求解带 Dirichlet 边界的 Poisson 方程 
        """)

parser.add_argument('--nx',
        default=5, type=int,
        help='初始笛卡尔网格 x 方向剖分段数, 默认 10 段.')

parser.add_argument('--ny',
        default=5, type=int,
        help='初始笛卡尔网格 y 方向剖分段数, 默认 10 段.')

parser.add_argument('--nz',
        default=5, type=int,
        help='初始笛卡尔网格 z 方向剖分段数, 默认 10 段.')

parser.add_argument('--maxit',
        default=3, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

nx = args.nx
ny = args.ny
nz = args.nz
maxit = args.maxit

pde = PDE()
box = pde.domain()
errorType = ['$|| u - u_h||_{\infty}$', '$|| u - u_h ||_{l_2}||$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float64)
ND = np.zeros(maxit, dtype=np.int_)
hs = np.zeros(maxit, dtype=np.float64)
for i in range(maxit):
    print("The {}-th computation:".format(i))
    # 0. 创建结构网格对象
    mesh = StructureHexMesh(box, nx=nx, ny=ny, nz=nz)
    hs[i] = max(mesh.hx, mesh.hy, mesh.hz)
    
    # 1. 组装矩阵和右端项 
    A = mesh.laplace_operator()
    F = mesh.interpolation(pde.source)
     
    # 2. 创建解向量数组, 内部节点对应的函数值初始化为 0，
    #    边界节点对应的函数值初始化 Dirichlet 边界条件的值
    NN = mesh.number_of_nodes()#节点个数
    ND[i] = NN
    uh = np.zeros(NN, dtype=np.float64)
    
    node = mesh.entity('node')
    isBdNode = mesh.ds.boundary_node_flag()
    uh[isBdNode] = pde.dirichlet(node[isBdNode]) 
    
    # 3. 处理 Dirichlet 边界条件

    # 3.1 处理右端
    F -= A@uh
    F[isBdNode] = uh[isBdNode]
    
    # 3.2 处理矩阵，边界点对应的行和列，除对角线元素外取 1 外， 其它值设为 0
    #     并保持矩阵的对称性
    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isBdNode] = 1
    Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    A = T@A@T + Tbd
    
    # 4. 求解
    uh[:] = spsolve(A, F)
    print(uh)
    
    # 4. 计算误差
    uI = mesh.interpolation(pde.solution)

    
    errorMatrix[0, i] = np.max(np.abs(uh - uI))
    errorMatrix[1, i] = np.sqrt(np.sum((uh - uI)**2)/NN)
    
    if i < maxit-1:
        nx *= 2
        ny *= 2
        nz *= 2

fig = plt.figure()
axes = fig.add_subplot()
showmultirate(axes, 0, ND, errorMatrix,  errorType, propsize=20)
plt.show()   
    
    
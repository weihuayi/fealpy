#!/usr/bin/env python3
# 
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.sparse import kron, eye, spdiags
from fealpy.solver.mg import MG
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh

from fealpy.tools.show import showmultirate, show_error_table
from fealpy.pde.poisson_2d import CosCosData as PDE


nx = int(sys.argv[1]) # 网格 x 方向剖分段数
ny = int(sys.argv[2]) # 网格 y 方向剖分段数
n = int(sys.argv[3]) # 初始网格的加密次数
maxit = int(sys.argv[4]) # 迭代加密的次数


def apply_boundary(mesh, pde, A, b):
    
    NN = mesh.number_of_nodes()
    node = mesh.entity('node')
    
    # deal with boundary condition
    isBDNode = mesh.ds.boundary_node_flag()
    idx, = np.nonzero(isBDNode)

    x = np.zeros(NN, dtype=mesh.ftype)
    x[idx] = pde.dirichlet(node[idx])
    #bnew = b.copy()

    bnew = b - A@x
    bnew[idx] = pde.dirichlet(node[idx])
    
    bdIdx = np.zeros((A.shape[0],), dtype=mesh.itype)
    bdIdx[idx] = 1

    Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
    T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[1])
    AD = T@A@T + Tbd

    return AD, bnew

 

pde = PDE() # 创建 pde 模型

# 误差类型与误差存储数组
errorType = ['$|| u - u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

# 自由度数组
Ndof = np.zeros(maxit, dtype=np.int)

# 创建初始网格对象 

for i in range(maxit):
    mesh = StructureQuadMesh(np.array([0,1,0,1]), nx, ny)
    PMatrix = mesh.uniform_refine(n=n, returnim=True)

    hx = mesh.hx
    hy = mesh.hy

    node = mesh.entity('node')
    uI = pde.solution(node)
    
    A = mesh.laplace_operator()       
    b = pde.source(node)   
    
    print('NC', mesh.number_of_cells())

    A, b = apply_boundary(mesh, pde, A, b)
    mg = MG(A, b, PMatrix, c=1/4)
    uh = mg.v_cycle()


    Ndof[i] = mesh.number_of_nodes()

    # 计算 L2 误差
    isBDNode = mesh.ds.boundary_node_flag()
    idx, = np.nonzero(~isBDNode)
    uL2 = np.sqrt(np.sum(hx*hy*(uI[idx] - uh[idx])**2))
    
    errorMatrix[0, i] = uL2 # 计算 L2 误差
    if i < maxit - 1:
        nx = 2*nx
        ny = 2*ny # 一致加密网格

# 显示误差
show_error_table(Ndof, errorType, errorMatrix)
# 可视化误差收敛阶
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()


#!/usr/bin/env python3
"""
3D弱有限元方法求解 poisson 方程
网格：四面体网格、
基函数：默认缩放单项式，其他暂无
各空间维数：k, k, k
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.functionspace.WeakGalerkinSpace3d import WeakGalerkinSpace3d
from WG3D_tet import WeakGalerkinSpace3d
from fealpy.pde.poisson_3d import CosCosCosData as PDE
from fealpy.mesh import MeshFactory as MF
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate

# solver
from scipy.sparse.linalg import spsolve

# 参数解析
parser = argparse.ArgumentParser(description="""多边形网格上任意次 3DWG 有限元方法""")

parser.add_argument('--degree',
                    default=1, type=int,
                    help=' WG 空间的次数, 默认为 1 次.')  #

parser.add_argument('--ns',
                    default=1, type=int,
                    help='初始网格部分段数, 默认 1 段.')

parser.add_argument('--maxit',
                    default=4, type=int,
                    help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args(args=[])

degree = args.degree
ns = args.ns
maxit = args.maxit

# 模型导入
pde = PDE()
box = [0, 1, 0, 1, 0, 1]

# 误差设置
errorType = ['$|| u - u_h||_{\\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\\Omega, 0}$']  # 作为图例解释

errorMatrix = np.zeros((2, maxit), dtype=np.float64)  # 用于存储每一次加密的两种误差
NDof = np.zeros(maxit, dtype=np.float64)  # 用于存储每一次加密后自由度的数量

# 加密求解
for i in range(maxit):
    mesh = MF.boxmesh3d(box=box, nx=ns, ny=ns, nz=ns, meshtype='tet')  # 四面体网格
    space = WeakGalerkinSpace3d(mesh, p=degree)  # WG空间建立
    NDof[i] = space.number_of_global_dofs()  # 获取自由度数量
    bc = DirichletBC(space, pde.dirichlet)  # 边界离散

    uh = space.function()  # 离散空间中的函数
    A = space.stiff_matrix()  # 刚度矩阵
    A += space.stabilizer_matrix()  # 稳定子矩阵
    F = space.source_vector(pde.source)  # 右端源项
    A, F = bc.apply(A, F, uh)  # 添加 Dirichlet 边界项

    uh[:] = spsolve(A, F).reshape(-1)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh,)  # 解的L2误差
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)  # 解的梯度的L2误差

    print(i)
    if i < maxit - 1:  # 加密 每次分段数加倍
        ns *= 2

# 绘图
fig = plt.figure()                                  # 建立画布
axes = fig.add_subplot(1, 1, 1, projection='3d')    # 3D子图
uh.add_plot(axes, cmap='rainbow')                   # 绘制数值解

fig = plt.figure()                                  # 新画布
axes = fig.add_subplot(projection='3d')                                  # 加入坐标
mesh.add_plot(axes)                                 # 绘制网格

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)  # 绘制对数误差图的接口 propsize:图例大小
plt.show()

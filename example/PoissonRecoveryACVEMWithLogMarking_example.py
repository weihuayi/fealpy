#!/usr/bin/env python3
# 

import argparse
import numpy as np

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        自适应方法协调虚单元方法求解奇性 Poisson 方程 
        """)


parser.add_argument('--pde',
        default='LShape', type=str,
        help='PDE 模型， 默认为 L 型区域上的 Poisson 方程.')

parser.add_argument('--degree',
        default=1, type=int,
        help='协调虚单元空间的次数, 默认为 1 次.')

parser.add_argument('--marker',
        default='log', type=str,
        help='标记策略类型，默认为 log 类型')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

pde = args.pde
degree = args.degree
marker = args.marker
maxit = args.maxit

if pde in {'LShape', 'LS'}:
    from fealpy.pde.poisson_2d import LShapeRSinData
    pde = LShapeRSinData()
    box = [-1, 1, -1, 1]
    mesh = MF.boxmesh2d(box, nx=10, ny=10, meshtype='quad')
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    node, cell = MF.delete_cell(node, cell, lambda p: p[..., 0] > 0 & p[..., 1] < 0)


errorType = [
        '$\| u_I - u_h \|_{l_2}$',
        '$\|\\nabla u_I - \\nabla u_h\|_A$',
        '$\| u - \Pi^\Delta u_h\|_0$',
        '$\|\\nabla u - \\nabla \Pi^\Delta u_h\|$',
        '$\eta$'
        ]

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
mesh = quadtree.to_pmesh()

for i in range(maxit):
    print('step:', i)
    vem = PoissonVEMModel(pde, mesh, p=p, q=q)
    vem.solve()
    eta = vem.recover_estimate(rtype='inv_area', residual=True)
    Ndof[i] = vem.space.number_of_global_dofs()
    errorMatrix[0, i] = vem.l2_error()
    errorMatrix[1, i] = vem.uIuh_error()
    errorMatrix[2, i] = vem.L2_error()
    if m == 1:
        errorMatrix[3, i] = vem.H1_semi_error_Kellogg()
    else:
        errorMatrix[3, i] = vem.H1_semi_error()

    errorMatrix[4, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        options = quadtree.adaptive_options()
        quadtree.adaptive(eta, options)
        mesh = quadtree.to_pmesh()

mesh.add_plot(plt, showaxis=True)

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.node[:, 0]
y = mesh.node[:, 1]
tri = quadtree.leaf_cell(celltype='tri')
axes.plot_trisurf(x, y, tri, vem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)

showmultirate(plt, k, Ndof, errorMatrix[2:, :], errorType[2:])
plt.show()

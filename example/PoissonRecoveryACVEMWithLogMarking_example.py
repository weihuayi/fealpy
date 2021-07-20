#!/usr/bin/env python3
# 

import argparse
import numpy as np
from scipy.sparse.linalg import spsolve

from fealpy.mesh import MeshFactory as MF
from fealpy.mesh import QuadrangleMesh, HalfEdgeMesh2d
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        自适应协调虚单元方法求解奇性 Poisson 方程 
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

parser.add_argument('--theta',
        default= 1.0, type=float,
        help='theta  参数, 默认为 1.0')

parser.add_argument('--maxdof',
        default=80000, type=int,
        help='默认网格自适应加密最大自由度个数, 默认最大自由度个数为 80000')

parser.add_argument('--plot',
        default=1, type=int,
        help='是否画图, 默认取值为 1， 画图 ')

args = parser.parse_args()

pde = args.pde
degree = args.degree
marker = args.marker
theta = args.theta
maxdof = args.maxdof

if pde in {'LShape', 'LS'}:
    from fealpy.pde.poisson_2d import LShapeRSinData
    pde = LShapeRSinData()
    box = [-1, 1, -1, 1]
    mesh = MF.boxmesh2d(box, nx=10, ny=10, meshtype='quad')
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    node, cell = MF.delete_cell(node, cell, 
            lambda p: (p[..., 0] > 0) & (p[..., 1] < 0))
    mesh = QuadrangleMesh(node, cell)
    mesh = HalfEdgeMesh2d.from_mesh(mesh)
    mesh.init_level_info()

errorType = [
        '$\\eta$',
        '$\| u - \Pi^\Delta u_h\|_0$',
        '$\|\\nabla u - \\nabla \Pi^\Delta u_h\|$',
        ]

NDof = [] 
errorMatrix = [[], [], []] 

k = 0
while True:

    if args.plot:
        fname = './test-' + str(k) + '.png'
        mesh.add_plot(plt)
        plt.savefig(fname)
        plt.close()

    space = ConformingVirtualElementSpace2d(mesh, p=degree)
    NDof += [space.number_of_global_dofs()]

    print("The {}-th computation with dof {}".format(k, NDof[-1]))

    uh = space.function()

    bc = DirichletBC(space, pde.dirichlet)

    A = space.stiff_matrix()
    F = space.source_vector(pde.source)
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F)

    sh = space.project_to_smspace(uh)
    eta = space.recovery_estimate(uh, pde, method='inv_area')

    #print("eta_init = ", eta)
    #eta = space.smooth_estimator(eta)
    #print("eta_smooth = ", eta)

    errorMatrix[0] += [np.sqrt(np.sum(eta**2))]
    errorMatrix[1] += [space.integralalg.error(pde.solution, sh.value)]
    errorMatrix[2] += [space.integralalg.error(pde.gradient, sh.grad_value)]

    if NDof[-1] < maxdof:
        if marker == 'log':
            options = mesh.adaptive_options(theta=theta, maxcoarsen=3)
            mesh.adaptive(eta, options)
        elif marker == 'L2':
            isMarkedCell = mesh.refine_marker(eta, theta, method='L2')
            mesh.refine_poly(isMarkedCell=isMarkedCell)
        k += 1
    else:
        break

NDof = np.array(NDof)
errorMatrix = np.array(errorMatrix)

showmultirate(plt, k-5, NDof, errorMatrix, errorType)
plt.show()

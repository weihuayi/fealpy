#!/usr/bin/env python3
# 

import time
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import LShapeRSinData
from fealpy.mesh import TriangleMesh 
from fealpy.mesh.halfedge_mesh import HalfEdgeMesh2d
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator 
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import LinearRecoveryAlg

from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showmultirate

import ipdb

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nx',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--theta',
        default=0.2, type=int,
        help='L2 标记策略中每次加密单元数量与总单元数量之比, 该数值在 0-1 之间越大代表每次加密的单元个数越多.')

parser.add_argument('--maxdof',
        default=200000, type=int,
        help='默认自适应过程中自由度最大个数, 默认值 20 万')

parser.add_argument('--maxit',
        default=30, type=int,
        help='默认自适应迭代最大个数，默认为 30 次')

args = parser.parse_args()

p = args.degree
nx = args.nx
ny = args.ny
theta = args.theta
maxit = args.maxit
maxdof = args.maxdof

pde = LShapeRSinData()
mesh = pde.init_mesh(n=4, meshtype='tri')
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3) # 使用半边网格

alg = LinearRecoveryAlg() # 梯度恢复算法

errorType = ['$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$']

NDof = np.zeros((maxit,), dtype=np.int_)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float64)

alluh = []

for i in range(maxit):
    t0 = time.time()
    print("The {}-th computation:".format(i))
    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator(q=p+2))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=p+2))
    F = lform.assembly()

    bc = DirichletBC(space, pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F)

    eta = alg.recovery_estimate(uh, method='harmonic')

    errorMatrix[0, i] = mesh.error(pde.solution, uh)
    errorMatrix[1, i] = mesh.error(pde.gradient, uh.grad_value)

    ## 若一个函数要插值，那么必须重新构造一个网格来复制这个函数
    mesh_copy = copy.deepcopy(mesh)
    space_copy = LagrangeFESpace(mesh_copy, p=p)
    uh_copy = space_copy.function()
    uh_copy[:] = uh[:]

    if len(alluh)>0:
        alluh = space_copy.interpolation_fe_function(alluh)
    ## 将函数插值到现在的空间中
    alluh.append(uh_copy)

    if (i < maxit-1)&(NDof[i]<maxdof):
        isMarkedCell = mark(eta, theta=theta)
        mesh.adaptive_refine(isMarkedCell, method='nvb')
    else:
        NDof = NDof[:i+1]
        errorMatrix = errorMatrix[:, :i+1]

# 以最后一个解作为真解，计算误差
errorMatrix0 = np.zeros_like(errorMatrix)
for i in range(len(alluh)):
    errorMatrix0[0, i] = mesh.error(alluh[i], alluh[-1])
    errorMatrix0[1, i] = mesh.error(alluh[i].grad_value, alluh[-1].grad_value)

print(errorMatrix)
print(errorMatrix0)
showmultirate(plt, maxit - 5, NDof, errorMatrix, errorType,
        propsize=40)
plt.show()

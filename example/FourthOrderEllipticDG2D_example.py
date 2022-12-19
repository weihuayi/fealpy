#!/usr/bin/env python3
# 

"""
IPDG 方法求解四阶问题, 

.. math::
    \Delta^2 u - \Delta u = f

"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from fealpy.pde.fourth_elliptic import PolynomialData, LShapeRSinData, SqrtData
from fealpy.mesh import MeshFactory
from fealpy.mesh import PolygonMesh
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.tools import showmultirate, show_error_table

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        二维多边形网格上 IPDG 方法求解 Poisson 方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='多项式空间的次数, 默认为 1 次.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--beta',
        default=1, type=float,
        help='TODO: 增加描述信息')

parser.add_argument('--alpha',
        default=1, type=float,
        help='TODO：增加描述信息')
args = parser.parse_args()

degree = args.degree
maxit = args.maxit
beta = args.beta
alpha = args.alpha 

# 误差类型与误差存储数组
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float64)
# 自由度数组
NDof = np.zeros(maxit, dtype=np.int_)

pde = PolynomialData()
#pde = LShapeRSinData(1)
pde = SqrtData()

for i in range(maxit):
    print("The {}-th computation:".format(i))
    mesh = pde.init_mesh(n=i+2, meshtype='tri') 
    mesh = PolygonMesh.from_mesh(mesh)
    space = ScaledMonomialSpace2d(mesh,degree)
    
    isInEdge = ~mesh.ds.boundary_edge_flag()
    isBdEdge = mesh.ds.boundary_edge_flag()

    #组装矩阵
    A = space.stiff_matrix()
    J = space.penalty_matrix(index=isInEdge)
    Q = space.normal_grad_penalty_matrix(index=isInEdge)
    S0 = space.flux_matrix(index=isInEdge)
    S1 = space.flux_matrix()

    A11 = A-S0-S1.T+alpha*J+beta*Q
    A12 = -space.mass_matrix()
    A22 = A11.T-A12
    A21 = alpha*space.penalty_matrix() 
    AD = bmat([[A11, A12], [A21, A22]], format='csr')

    #组装右端向量
    F11 = space.edge_source_vector(pde.gradient, index=isBdEdge, hpower=0)
    F12 = -space.edge_normal_source_vector(pde.dirichlet, index=isBdEdge)
    F21 = space.edge_source_vector(pde.dirichlet, index=isBdEdge)
    F22 = space.source_vector0(pde.source)
    F = np.r_[F11+F12, F21+F22]

    #求解
    gdof = space.number_of_global_dofs(p=degree)
    uh = space.function()
    uh[:] = spsolve(AD, F)[:gdof]

    NDof[i] = space.number_of_global_dofs()
    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh, power=2)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value,
            power=2)
    if i > 0:
        print("order = ", np.log(errorMatrix[0, i]/errorMatrix[0, i-1])/np.log(NDof[i-1]/NDof[i]))
    
# 显示误差
show_error_table(NDof, errorType, errorMatrix)

# 可视化误差收敛阶
showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=12)

plt.show()


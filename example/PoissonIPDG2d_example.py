#!/usr/bin/env python3
# 

"""
IPDG 方法求解二维 Poisson 方程, 

.. math::
    -\Delta u = f

"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import PolynomialData as PDE
from fealpy.mesh import MeshFactory
from fealpy.mesh import PolygonMesh
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.tools import showmultirate, show_error_table

from PoissonIPDGModel2d import PoissonIPDGModel2d

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
        default=200, type=float,
        help='TODO: 增加描述信息')

parser.add_argument('--alpha',
        default=-1, type=float,
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
pde = PDE()
for i in range(maxit):
    print("The {}-th computation:".format(i))
    mesh = pde.init_mesh(n=i+3, meshtype='tri') 
    mesh = PolygonMesh.from_mesh(mesh)
    space = ScaledMonomialSpace2d(mesh,degree)
    NDof[i] = space.number_of_global_dofs()
    
    model = PoissonIPDGModel2d(pde, mesh, degree) # 创建 Poisson IPDG 有限元模型
    model.solve(beta, alpha) # 求解
    uh = model.uh
    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh, power=2)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value,
            power=2)
    
# 显示误差
show_error_table(NDof, errorType, errorMatrix)
# 可视化误差收敛阶
showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=12)

plt.show()



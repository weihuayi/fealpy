#!/usr/bin/env python3
# 

import time
import sys
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import SecondNedelecFESpace
from fealpy import logger
logger.setLevel('WARNING')

from fealpy.backend import backend_manager as bm

# 双线性型
from fealpy.fem import BilinearForm

# 线性型
from fealpy.fem import LinearForm

# 积分子
from fealpy.fem import CurlCurlIntegrator
from fealpy.fem import VectorSourceIntegrator,ScalarMassIntegrator
from fealpy.fem import DirichletBC

from fealpy.decorator import cartesian, barycentric
from fealpy.tools.show import showmultirate, show_error_table

# solver
from fealpy.solver import spsolve

from fealpy.pde.maxwell_2d import SinData as PDE2d
from fealpy.pde.maxwell_3d import BubbleData3d as PDE3d
from fealpy.utils import timer


# 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次有限元方法求解possion方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

parser.add_argument('--dim',
        default=2, type=int,
        help='默认维数为2')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认最大迭代次数为4')

args = parser.parse_args()
p = args.degree
maxit = args.maxit
dim = args.dim
backend = args.backend
bm.set_backend(backend)
if dim == 2:
    pde = PDE2d()
else:
    pde = PDE3d()

errorType = ['$|| E - E_h||_0$ with k=2',
             '$|| E - E_h||_0$ with k=3',
             '$|| E - E_h||_0$ with k=4',
             '$||\\nabla \\times u - \\nabla_h \\times u_h||_0$ with k=2',
             '$||\\nabla \\times u - \\nabla_h \\times u_h||_0$ with k=3',
             '$||\\nabla \\times u - \\nabla_h \\times u_h||_0$ with k=4']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.float64)

tmr = timer()
next(tmr)

ps = [2, 3, 4]
#ps = [2,3]
for j, p in enumerate(ps):
    for i in range(maxit):
        print("The {}-th computation:".format(i))

        if dim == 2:
            mesh = TriangleMesh.from_box(pde.domain(), nx=2**i, ny=2**i) 
            space = SecondNedelecFESpace(mesh, p=p)
        else:
            mesh = TetrahedronMesh.from_box(pde.domain(), nx=2**i, ny=2**i, nz=2**i)
            space = SecondNedelecFESpace(mesh, p=p)
        tmr.send(f'第{i}次网格和pde生成时间')

        gdof = space.dof.number_of_global_dofs()
        NDof[i] = 1/2**i 

        bform = BilinearForm(space)
        #bform.add_integrator(VectorMassIntegrator(coef=-1, q=p+3))
        bform.add_integrator(ScalarMassIntegrator(coef=-1, q=p+3))
        bform.add_integrator(CurlCurlIntegrator(coef=1, q=p+3))
        A = bform.assembly()
        tmr.send(f'第{i}次矩组装时间')

        lform = LinearForm(space)
        lform.add_integrator(VectorSourceIntegrator(pde.source, q=p+3))
        F = lform.assembly()
        tmr.send(f'第{i}次向量组装时间')

        # Dirichlet 边界条件
        Eh = space.function()
        bc = DirichletBC(space, pde.dirichlet)
        A, F = bc.apply(A, F)
        tmr.send(f'第{i}次边界处理时间')

        #Eh[:] = cg(A, F, maxiter=5000, atol=1e-14, rtol=1e-14)
        Eh[:] = spsolve(A, F,"scipy")
        tmr.send(f'第{i}次求解器时间')

        # 计算误差
        errorMatrix[j, i] = mesh.error(pde.solution, Eh.value)
        errorMatrix[j+3, i] = mesh.error(pde.curl_solution, Eh.curl_value)
        tmr.send(f'第{i}次误差计算及网格加密时间')
        print(errorMatrix)

next(tmr)
showmultirate(plt, 2, NDof, errorMatrix,  errorType, propsize=20)
show_error_table(NDof, errorType, errorMatrix)
plt.show()


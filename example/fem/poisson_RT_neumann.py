#!/usr/bin/env python3

import argparse 
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import RaviartThomasFESpace
from fealpy.functionspace import LagrangeFESpace
from fealpy import logger
logger.setLevel('WARNING')

from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm,ScalarMassIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator,BoundaryFaceSourceIntegrator
from fealpy.fem import DivIntegrator
from fealpy.fem import BlockForm,LinearBlockForm

# from fealpy.decorator import cartesian, barycentric
from fealpy.tools.show import showmultirate, show_error_table

# solver
from fealpy.solver import spsolve

from fealpy.pde.poisson_2d import CosCosData as PDE2d
from fealpy.pde.poisson_3d import CosCosCosData as PDE3d
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


errorType = ['$|| p - p_h||_0$ with k=2',
             '$|| p - p_h||_0$ with k=3',
             '$|| p - p_h||_0$ with k=4',
             '$|| u - u_h||_0$ with k=2',
             '$|| u - u_h||_0$ with k=3',
             '$|| u - u_h||_0$ with k=4']
errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
NDof = bm.zeros(maxit, dtype=bm.float64)

tmr = timer()
next(tmr)

ps = [2, 3, 4]
for j, p in enumerate(ps):
    for i in range(maxit):
        print("The {}-th computation:".format(i))
        
        if dim == 2:
            mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2**i, ny=2**i)
            space1 = LagrangeFESpace(mesh,p=p,ctype="D")
            space2 = RaviartThomasFESpace(mesh, p=p)
        else:
            mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=2**i, ny=2**i, nz=2**i)
            space1 = LagrangeFESpace(mesh,p=p,ctype="D")
            space2 = RaviartThomasFESpace(mesh, p=p)
        tmr.send(f'第{i}次网格和pde生成时间')

        pdof = space1.dof.number_of_global_dofs()
        udof = space2.dof.number_of_global_dofs()
        NDof[i] = 1/2**i

        uh = space2.function()
        ph = space1.function()

        bform1 = BilinearForm(space2)
        bform1.add_integrator(ScalarMassIntegrator(coef=1, q=p+3))

        bform2 = BilinearForm((space1,space2))
        bform2.add_integrator(DivIntegrator(coef=-1, q=p+3))

        M = BlockForm([[bform1,bform2],
                       [bform2.T,None]])
        M = M.assembly()
        tmr.send(f'第{i}次矩组装时间')
        # 组装右端

        lform = LinearForm(space1)
        lform.add_integrator(ScalarSourceIntegrator(pde.source))
        F = lform.assembly()
        G = space2.set_neumann_bc(pde.solution)
        b = bm.concatenate([-G,-F],axis=0)
        tmr.send(f'第{i}次向量组装时间')

        val = spsolve(M, b,"scipy")

        uh[:] = val[:udof]
        ph[:] = val[udof:]
        tmr.send(f'第{i}次求解时间')

        #计算误差
        errorMatrix[j, i] = mesh.error(pde.solution, ph)
        errorMatrix[j+3, i] = mesh.error(pde.flux, uh.value)
        print("error = ", errorMatrix)

next(tmr)
showmultirate(plt, 2, NDof, errorMatrix,  errorType, propsize=20)
show_error_table(NDof, errorType, errorMatrix)
plt.show()
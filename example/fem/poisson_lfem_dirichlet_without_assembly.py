#!/usr/bin/python3
#import ipdb
import argparse
from matplotlib import pyplot as plt

from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')
from fealpy.backend import backend_manager as bm



## 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次有限元方法求解possion方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--n',
        default=4, type=int,
        help='初始网格剖分段数，默认每个方向剖分 4 段')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy.")

parser.add_argument('--meshtype',
        default='tri', type=str,
                    help="默认网格为 tri (三角形网格)"
                         "int: 区间网格；tri：三角形网格；"
                         "quad: 四边形网格；tet: 四面体网格"
                         "hex: 六面体网格"
                    )

args = parser.parse_args()
bm.set_backend(args.backend)

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBCOperator
from fealpy.solver import cg

p = args.degree
n = args.n
meshtype = args.meshtype
maxit = args.maxit

tmr = timer()
next(tmr)
if meshtype == 'int':
    from fealpy.pde.poisson_1d import CosData 
    from fealpy.mesh import IntervalMesh
    pde = CosData()
    mesh = IntervalMesh.from_interval_domain([0,1], n)
elif meshtype == 'tri':
    from fealpy.pde.poisson_2d import CosCosData 
    from fealpy.mesh import TriangleMesh
    pde = CosCosData()
    mesh = TriangleMesh.from_box([0,1,0,1], n, n)
elif meshtype == 'quad':
    from fealpy.pde.poisson_2d import CosCosData 
    from fealpy.mesh import QuadrangleMesh
    pde = CosCosData()
    mesh = QuadrangleMesh.from_box([0,1,0,1], n, n)
elif meshtype == 'tet':
    from fealpy.pde.poisson_3d import CosCosCosData 
    from fealpy.mesh import TetrahedronMesh
    pde = CosCosCosData()
    mesh = TetrahedronMesh.from_box([0,1,0,1,0,1], n, n, n)
elif meshtype == 'hex':
    from fealpy.pde.poisson_3d import CosCosCosData 
    from fealpy.mesh import HexahedronMesh
    pde = CosCosCosData()
    mesh = HexahedronMesh.from_box([0,1,0,1,0,1], n, n, n)
else: 
    raise ValueError(f"Unsupported : {meshtype} mesh")

errorType = ['$|| u - u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
tmr.send('网格和pde生成时间')


for i in range(maxit):
    space= LagrangeFESpace(mesh, p=p)
    tmr.send(f'第{i}次空间时间') 

    uh = space.function() # 建立一个有限元函数

    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator())
    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(pde.source))

    F = lform.assembly()
    tmr.send(f'第{i}次矩组装时间')

    #ipdb.set_trace()
    bcop = DirichletBCOperator(bform, gd=pde.dirichlet)
    u0 = bcop.init_solution()
    F = bcop.apply(F, u0)
    tmr.send(f'第{i}次边界处理时间')

    uh[:] = cg(bcop, F, x0=u0, maxiter=5000, atol=1e-14, rtol=1e-14)
    tmr.send(f'第{i}次求解器时间')

    errorMatrix[0, i] = mesh.error(pde.solution, uh)

    if i < maxit-1:
        mesh.uniform_refine(n=1)
    tmr.send(f'第{i}次误差计算及网格加密时间')
next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))

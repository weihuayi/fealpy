#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: poisson_lfem_robin.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 28 Oct 2024 08:01:29 PM CST
	@bref 
	@ref 
'''  
import argparse
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.utils import timer
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem.face_mass_integrator import BoundaryFaceMassIntegrator 
from fealpy.fem.face_source_integrator import BoundaryFaceSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.pde.poisson_2d import CosCosData 
from fealpy.mesh import TriangleMesh, QuadrangleMesh
from fealpy.mesh import Mesh
from fealpy.solver import cg


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次有限元方法求解半线性方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--n',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等')

parser.add_argument('--meshtype',
        default='tri', type=str,
                    help="默认网格为 tri (三角形网格)"
                         "tri：三角形网格；quad: 四边形网格"
                    )

args = parser.parse_args()
args.backend = 'pytorch'
bm.set_backend(args.backend)
p = args.degree
n = args.n
meshtype = args.meshtype
maxit = args.maxit

tmr = timer()
next(tmr)
if meshtype == 'tri':
    from fealpy.pde.poisson_2d import CosCosData 
    from fealpy.mesh import TriangleMesh
    pde = CosCosData()
    mesh = TriangleMesh.from_box([0,1,0,1], n, n)
elif meshtype == 'quad':
    from fealpy.pde.poisson_2d import CosCosData 
    from fealpy.mesh import QuadrangleMesh
    pde = CosCosData()
    mesh = QuadrangleMesh.from_box([0,1,0,1], n, n)
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
    bform.add_integrator(BoundaryFaceMassIntegrator(coef=pde.kappa, threshold=pde.is_robin_boundary))
    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(pde.source))
    lform.add_integrator(BoundaryFaceSourceIntegrator(source=pde.robin, threshold=pde.is_robin_boundary))
    lform.add_integrator(BoundaryFaceSourceIntegrator(source=pde.neumann, threshold=pde.is_neumann_boundary))
    
    A = bform.assembly()
    F = lform.assembly()
    tmr.send(f'第{i}次矩组装时间')

    gdof = space.number_of_global_dofs()

    A, F = DirichletBC(space, gd=pde.dirichlet, threshold=pde.is_dirichlet_boundary).apply(A, F)
    tmr.send(f'第{i}次边界处理时间')

    uh[:] = cg(A, F, maxit=5000, atol=1e-14, rtol=1e-14)
    tmr.send(f'第{i}次求解器时间')

    errorMatrix[0, i] = mesh.error(pde.solution, uh)

    if i < maxit-1:
        mesh.uniform_refine(n=1)
    tmr.send(f'第{i}次误差计算及网格加密时间')

next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))

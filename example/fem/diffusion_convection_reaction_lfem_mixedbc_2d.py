#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

# 模型数据
from fealpy.pde.diffusion_convection_reaction_2d import PDEData_0 as PDE

# 拉格朗日有限元空间
from fealpy.functionspace import LagrangeFESpace

# 区域积分子
from fealpy.fem import ScalarDiffusionIntegrator      # (A\nabla u, \nabla v) 
from fealpy.fem import ScalarConvectionIntegrator     # (b\cdot \nabla u, v) 
from fealpy.fem import ScalarMassIntegrator           # (r*u, v)
from fealpy.fem import ScalarSourceIntegrator         # (f, v)

# 边界积分子
from fealpy.fem import ScalarNeumannSourceIntegrator  # <g_N, v>
from fealpy.fem import ScalarRobinSourceIntegrator    # <g_R, v>
from fealpy.fem import ScalarRobinBoundaryIntegrator  # <kappa*u, v>

# 双线性型
from fealpy.fem import BilinearForm

# 线性型
from fealpy.fem import LinearForm

# 第一类边界条件
from fealpy.fem import DirichletBC


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh \ QuadrangleMesh 上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--mtype',
        default='tri', type=str,
        help='网格类型 tri 或者 quad, 默认为 tri.')

parser.add_argument('--nx',
        default=8, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=8, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

p = args.degree
mtype = args.mtype
nx = args.nx
ny = args.ny
maxit = args.maxit

pde = PDE(kappa=1.0)
domain = pde.domain()

if mtype == 'tri':
    from fealpy.mesh import TriangleMesh
    mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
elif mtype == 'quad':
    from fealpy.mesh import QuadrangleMesh
    mesh = QuadrangleMesh.from_box(domain, nx=nx, ny=ny)


errorType = ['$|| u - u_h||_{\Omega,0}$', 
        '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    bform = BilinearForm(space)
    # (A(x)\nabla u, \nabla v)
    D = ScalarDiffusionIntegrator(q=p+3)
    # (b\cdot \nabla u, v)
    C = ScalarConvectionIntegrator(c=pde.convection_coefficient, q=p+3)
    # (r*u, v)
    M = ScalarMassIntegrator(q=p+3)
    # <kappa*u, v>
    R = ScalarRobinBoundaryIntegrator(pde.kappa,
            threshold=pde.is_robin_boundary, q=p+2)
    bform.add_domain_integrator([D, C, M]) 
    bform.add_boundary_integrator(R) 
    A = bform.assembly()

    lform = LinearForm(space)
    # (f, v)
    Vs = ScalarSourceIntegrator(pde.source, q=p+2)
    # <g_N, v>
    Vn = ScalarNeumannSourceIntegrator(pde.neumann, threshold=pde.is_neumann_boundary, q=p+2)
    # <g_R, v>
    Vr = ScalarRobinSourceIntegrator(pde.robin, threshold=pde.is_robin_boundary, q=p+2)
    lform.add_domain_integrator(Vs)
    lform.add_boundary_integrator([Vr, Vn])
    F = lform.assembly()

    # Dirichlet 边界条件
    bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = mesh.error(pde.solution, uh, q=p+2)
    errorMatrix[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+2)

    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

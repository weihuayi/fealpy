#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

import numpy as np
from fealpy.pde.diffusion_convection_reaction import NonConservativeDCRPDEModel2d

# 三角形网格
from fealpy.mesh import TriangleMesh

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
        TriangleMesh 上线性有限元求解对流占优问题
        """)

parser.add_argument('--solution',
        default='cos(pi*x)*cos(pi*y)', type=str,
        help='方程真解，默认为 cos(pi*x)*cos(pi*y)')

parser.add_argument('--dcoef',
        default='1', type=str,
        help='扩散项系数，默认为 1')

parser.add_argument('--ccoef',
        default=('1', '1'), nargs=2, type=str,
        help='对流项系数，默认为 (1, 1)')

parser.add_argument('--rcoef',
        default='1', type=str,
        help='反应项系数，默认为 1')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()
u = args.solution
A = args.dcoef
b = args.ccoef
c = args.rcoef
maxit = args.maxit
p = 1

# 强对流、弱扩散的例子
pde = NonConservativeDCRPDEModel2d(u=u, A=A, b=b, c=c) 
domain = pde.domain()

errorType = ['$|| u - u_h||_{\Omega,0}$', 
        '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)

D = ScalarDiffusionIntegrator(c=pde.diffusion_coefficient, q=p+3)
C = ScalarConvectionIntegrator(c=pde.convection_coefficient, q=p+3)
M = ScalarMassIntegrator(c=pde.reaction_coefficient, q=p+3)
f = ScalarSourceIntegrator(pde.source, q=p+3)

for i in range(maxit):
    mesh = TriangleMesh.from_box(domain, nx=10*2**i, ny=10*2**i)
    space = LagrangeFESpace(mesh, p=p)

    b = BilinearForm(space)
    b.add_domain_integrator([D, C, M]) 

    l = LinearForm(space)
    l.add_domain_integrator(f)

    A = b.assembly() 
    F = l.assembly()

    bc = DirichletBC(space, pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh) 
    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = mesh.error(pde.solution, uh, q=p+2)
    errorMatrix[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+2)

    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

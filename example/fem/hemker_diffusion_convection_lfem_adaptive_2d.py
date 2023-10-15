#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

import numpy as np
from fealpy.pde.diffusion_convection_reaction import  HemkerDCRModelWithBoxHole2d as PDE

# 三角形网格
from fealpy.mesh import TriangleMesh
from fealpy.mesh.adaptive_tools import mark

# 拉格朗日有限元空间
from fealpy.functionspace import LagrangeFESpace

# 区域积分子
from fealpy.fem import ScalarDiffusionIntegrator      # (A\nabla u, \nabla v) 
from fealpy.fem import ScalarConvectionIntegrator     # (b\cdot \nabla u, v) 
from fealpy.fem import ScalarPGLSConvectionIntegrator
from fealpy.fem import ScalarMassIntegrator           # (r*u, v)
from fealpy.fem import ScalarSourceIntegrator         # (f, v)

# 边界积分子
from fealpy.fem import ScalarNeumannSourceIntegrator  # <g_N, v>
from fealpy.fem import ScalarRobinSourceIntegrator    # <g_R, v>
from fealpy.fem import ScalarRobinBoundaryIntegrator  # <kappa*u, v>

from fealpy.fem import LinearRecoveryAlg

# 双线性型
from fealpy.fem import BilinearForm

# 线性型
from fealpy.fem import LinearForm

# 第一类边界条件
from fealpy.fem import DirichletBC

import ipdb

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上线性有限元求解对流占优问题
        """)

parser.add_argument('--order',
        default=1, type=int,
        help='有限元空间的次数，默认为 1 次')

parser.add_argument('--dcoef',
        default=1.0, type=float,
        help='扩散项系数，默认为 1.0')

parser.add_argument('--ccoef',
        default=(1.0, 0.0), nargs=2, type=float,
        help='对流项系数，默认为 (1, 0)')

parser.add_argument('--pgls',
        default=True, type=bool,
        help='是否采用 PGLS 方法， 默认采用')

parser.add_argument('--maxit',
        default=50, type=int,
        help='自适应迭代的次数，默认 50 次迭代')

parser.add_argument('--theta',
        default=0.2, type=float,
        help='自适应迭代参数，默认 0.2')


args = parser.parse_args()
p = args.order
A = args.dcoef
b = args.ccoef
pgls = args.pgls
maxit = args.maxit
theta = args.theta

# 强对流、弱扩散的例子
pde = PDE(A=A, b=b) 
domain = pde.domain()

D = ScalarDiffusionIntegrator(c=pde.diffusion_coefficient, q=p+3)

if pgls:
    C = ScalarPGLSConvectionIntegrator(A, np.array(b))
else: # 否则用有限元
    C = ScalarConvectionIntegrator(c=pde.convection_coefficient, q=p+3)

f = ScalarSourceIntegrator(pde.source, q=p+3)

mesh = TriangleMesh.from_box(domain.box, nx=60, ny=30, threshold= lambda p: pde.fd(p) < 0.0)

alg = LinearRecoveryAlg()

for i in range(maxit):

    space = LagrangeFESpace(mesh, p=p)

    b = BilinearForm(space)
    b.add_domain_integrator([D, C]) 

    l = LinearForm(space)
    l.add_domain_integrator(f)

    A = b.assembly() 
    F = l.assembly()

    bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh) 
    uh[:] = spsolve(A, F)
    print(uh[uh < 0.0])
    print(i, ": ", "is there exsit value smaller than 0.0?", np.any(uh < 0.0))
    eta = alg.recovery_estimate(uh, method='harmonic')
    if i < maxit - 1:
        isMarkedCell = mark(eta, theta=theta)
        mesh.bisect(isMarkedCell, options={'disp':False})
        mesh.add_plot(plt)
        plt.savefig('./test-' + str(i+1) + '.png')
        plt.close()

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)

# viridis、plasma、inferno、magma
mesh.show_function(plt, uh, cmap='rainbow')
plt.show()

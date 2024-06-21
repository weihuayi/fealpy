
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.pde.helmholtz_2d import HelmholtzData2d

#三角形网格
from fealpy.np.mesh import TriangleMesh

# 拉格朗日有限元空间
from fealpy.np.functionspace import LagrangeFESpace

#区域积分子
from fealpy.np.fem import ScalarDiffusionIntegrator      # (A\nabla u, \nabla v)
from fealpy.np.fem import ScalarMassIntegrator           # (r*u, v)
from fealpy.np.fem import ScalarSourceIntegrator         # (f, v)

#边界积分子
from fealpy.np.fem import ScalarRobinSourceIntegrator    # <g_R, v>
from fealpy.np.fem import ScalarRobinBoundaryIntegrator  # <kappa*u, v>

#双线性形
from fealpy.np.fem import BilinearForm

#线性形
from fealpy.np.fem import LinearForm

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上任意次有限元方法求解二维 Helmholtz 方程 
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--wavenum', 
        default=1, type=int,
        help='模型的波数, 默认为 1.')

parser.add_argument('--cip', nargs=2,
        default=[0, 0], type=float,
        help=' CIP-FEM 的系数, 默认取值 0, 即标准有限元方法.')

parser.add_argument('--ns',
        default=20, type=int,
        help='初始网格 x 和 y 方向剖分段数, 默认 20 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()
k = args.wavenum
kappa = k * 1j
c = complex(args.cip[0], args.cip[1])
ns = args.ns
maxit = args.maxit
p=1

pde = HelmholtzData2d(k=k) 
domain = pde.domain()

errorType = ['$|| u - u_I||_{\Omega,0}$',
             '$|| \\nabla u - \\nabla u_I||_{\Omega, 0}$',
             '$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$',
             ]

errorMatrix = np.zeros((4, maxit), dtype=np.float64)

D = ScalarDiffusionIntegrator()
M = ScalarMassIntegrator(-k**2)
R = ScalarRobinBoundaryIntegrator(kappa)
f = ScalarSourceIntegrator(pde.source)

Vr = ScalarRobinSourceIntegrator(pde.robin)


n = 64
mesh = TriangleMesh.from_box(domain, nx=n, ny=n)
mesh.node.astype(complex)
space = LagrangeFESpace(mesh, p=p)

b = BilinearForm(space)
b.add_integrator([D, M])
b.add_integrator(R)

l = LinearForm(space)
l.add_integrator([f, Vr])

A = b.assembly() 
F = l.assembly()

uh = space.function(dtype=np.complex128)
uh[:] = spsolve(A, F)

uI = space.interpolate(pde.solution)

bc = np.array([1/3, 1/3, 1/3])
ps = mesh.bc_to_point(bc)
u = pde.solution(ps)
uI = uI(bc)
uh = uh(bc)

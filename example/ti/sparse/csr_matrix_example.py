#!/usr/bin/env python3
# 

import argparse

import numpy as np
from numpy.linalg import norm
import taichi as ti

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarLaplaceIntegrator
from fealpy.fem import BilinearForm

from fealpy.ti.sparse import CSRMatrix

from fealpy.utils import timer

arch_map = {'cpu': ti.cpu, 'gpu':ti.cuda}

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        基于 Taichi 实现的 CSRMatrix 例子测试 
        """)

parser.add_argument('--p',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--n',
        default=100, type=int,
        help='初始网格 x, y 方向剖分段数.')


parser.add_argument('--arch',
        default='cpu', type=str,
        help='计算后端，默认 cpu, 可选 gpu')

args = parser.parse_args()

p = args.p 
n = args.n
arch = arch_map[args.arch]

ti.init(arch=arch)
n = 1000
p = 3

t = timer()
next(t)
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=n, ny=n)
space = LagrangeFESpace(mesh, p=p)
gdof = space.number_of_global_dofs()
form = BilinearForm(space)
form.add_domain_integrator(ScalarLaplaceIntegrator(q=p+3))
A = form.assembly()
v = np.ones(gdof, dtype=A.dtype)
t.send('prepare scipy csr_matrix')


B = CSRMatrix.from_scipy(A)
w = ti.field(ti.f64, shape=(gdof, ))
w.fill(1.0)
t.send('copy from scipy csr_matrix')

x = A@v
t.send('matrix multiply vector in scipy')

y = B@w
t.send('matrix multiply vector in fealpy.ti.sparse')

e = norm(x - y.to_numpy())
print(f'compute difference as {e}')
t.send(f'compute difference as {e}')
next(t)






#!/usr/bin/env python3
#

"""
混合元求解 Poisson 方程, 

.. math::
    -\Delta u = f

转化为

.. math::
    (\mathbf u, \mathbf v) - (p, \\nabla\cdot \mathbf v) &= - <p, \mathbf v\cdot n>_{\Gamma_D}
           - (\\nabla\cdot\mathbf u, w) &= - (f, w), w \in L^2(\Omega)

"""

import sys

import matplotlib.pyplot as plt

import numpy as np
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d


p = int(sys.argv[1]) # RT 空间的次数
n = int(sys.argv[2]) # 初始网格部分段数
maxit = int(sys.argv[3]) # 迭代求解次数

pde = CosCosData()  # pde 模型
box = pde.domain()  # 模型区域
mf = MeshFactory() # 网格工场

for i in range(maxit)
    mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype='tri')

    space = RaviartThomasFiniteElementSpace2d(mesh, p=p)

    udof = space.number_of_global_dofs()
    pdof = space.smspace.number_of_global_dofs()
    gdof = udof + pdof

    uh = space.function()
    ph = space.smspace.function()

    A = space.stiff_matrix()
    B = space.div_matrix()

    F0 = -space.set_neumann_bc(pde.dirichlet) # Poisson 的 D 氏边界变为 Neumann
    F1 = -space.source_vector(pde.source)

    AA = bmat([[A, -B], [-B.T, None]], format='csr')
    FF = np.r_['0', F0, F1]
    x = spsolve(AA, FF).reshape(-1)
    uh[:] = x[:udof]
    ph[:] = x[udof:]
    error0 = space.integralalg.error(pde.flux, uh, power=2)

    def f(bc):
        xx = mesh.bc_to_point(bc)
        return (pde.solution(xx) - ph(xx))**2
    error1 = space.integralalg.integral(f)

    n *= 2 # 加密网格 

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, box=box)
node = ps.reshape(-1, 2)
uv = uh.reshape(-1, 2)
axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1])
plt.show()

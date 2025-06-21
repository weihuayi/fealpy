#!/usr/bin/python3
import ipdb
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import barycentric
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')

from fealpy.backend import backend_manager as bm

from fealpy.pde.poisson_2d import CosCosData 
from fealpy.mesh import TriangleMesh
from fealpy.fem import PoissonLFEMSolver
from fealpy.solver.mumps import spsolve_triangular

from mpi4py import MPI # 提前导入




tmr = timer()
next(tmr)

p = 1
n = 3 
pde = CosCosData() 
domain = pde.domain()
mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
mesh.uniform_refine(n=5)
s0 = PoissonLFEMSolver(pde, mesh, p, timer=tmr, logger=logger)

r = s0.b
L = s0.A.tril()
print(L)
tmr.send('取下三角矩阵时间')
U = s0.A.triu()
tmr.send('取上三角矩阵时间')

x0 = spsolve_triangular(L, r, transpose=False, silent=False)
tmr.send('求解下三角系统时间')
residual = bm.max(bm.abs(L@x0 - r))
print(residual)

x1 = spsolve_triangular(L, r, transpose=True, silent=False)
tmr.send('求解上三角系统时间')
residual = bm.max(bm.abs(L.T@x1 - r))
print(residual)

x2 = spsolve_triangular(U, r, transpose=True, silent=False)
tmr.send('求解下三角系统时间')
residual = bm.max(bm.abs(U.T@x2 - r))
print(residual)

x3 = spsolve_triangular(U, r, transpose=False, silent=False)
tmr.send('求解上三角系统时间')
residual = bm.max(bm.abs(U@x3 - r))
print(residual)

uh = s0.solve()
tmr.send('求解离散系统时间')
next(tmr)

if False:
# Plot
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    fig = plt.figure()
    axes = fig.add_subplot(121)
    mesh.add_plot(axes)
    axes = fig.add_subplot(122, projection='3d')
    axes.plot_trisurf(node[:, 0], node[:, 1], uh, triangles=cell, cmap='rainbow')
    tmr.send('画图时间')
    plt.show()

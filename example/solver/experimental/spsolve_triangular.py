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
from fealpy.solver import spsolve_triangular




tmr = timer()
next(tmr)

p = 1
n = 3 
pde = CosCosData() 
domain = pde.domain()
mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
mesh.uniform_refine(n=4)
ipdb.set_trace()
s0 = PoissonLFEMSolver(pde, mesh, p, timer=tmr, logger=logger)

r = s0.b
L = s0.A.tril()
U = s0.A.triu()

print('cell.dtype', mesh.entity('cell').dtype)
print('mesh.itype', mesh.itype)
print('space.itype', s0.space.itype)
print('A.itype', s0.A.itype)
print('L.itype', L.itype)

x = spsolve_triangular(L, r, lower=True)
residual = bm.max(bm.abs(L@x - r))
print(residual)


uh = s0.solve()
# Plot
node = mesh.entity('node')
cell = mesh.entity('cell')
fig = plt.figure()
axes = fig.add_subplot(121)
mesh.add_plot(axes)
axes = fig.add_subplot(122, projection='3d')
axes.plot_trisurf(node[:, 0], node[:, 1], uh, triangles=cell, cmap='rainbow')
plt.show()

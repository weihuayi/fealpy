#!/usr/bin/python3
# import ipdb
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




tmr = timer()
next(tmr)

p = 1
n = 2
pde = CosCosData() 
domain = pde.domain()
mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)

IM = mesh.uniform_refine(n=3, returnim=True)

s0 = PoissonLFEMSolver(pde, mesh, p, timer=tmr, logger=logger)

# s0.cg_solve()
s0.gs_solve()
# s0.jacobi_solve()
# s0.gamg_solve(IM)
error0 = s0.L2_error()
print('error0:', error0)
s0.show_mesh_and_solution()


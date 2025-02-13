#!/usr/bin/python3
# import ipdb
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import barycentric
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('INFO')

from fealpy.backend import backend_manager as bm

from fealpy.pde.poisson_2d import CosCosData 
from fealpy.mesh import TriangleMesh
from fealpy.fem import PoissonLFEMSolver




tmr = timer()
next(tmr)

p = 3
n = 3
pde_2d = CosCosData() 
domain_2d = pde_2d.domain()
mesh_2d = TriangleMesh.from_box(box=domain_2d, nx=n, ny=n)

IM_2d = mesh_2d.uniform_refine(n=4, returnim=True)

s0_2d = PoissonLFEMSolver(pde_2d, mesh_2d, p, timer=tmr, logger=logger)

x,info = s0_2d.gamg_solve(IM_2d)
err = s0_2d.L2_error()
#展示最终解
s0_2d.show_mesh_and_solution()


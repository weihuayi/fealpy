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
from fealpy.pde.poisson_1d import CosData
from fealpy.mesh import TriangleMesh,IntervalMesh
from fealpy.fem import PoissonLFEMSolver




tmr = timer()
next(tmr)

p = 1
n = 2
pde = CosData()
domain = pde.domain()
mesh = IntervalMesh.from_interval_domain(interval=domain,nx=n)

IM = mesh.uniform_refine(n=4, returnim=True)

s0 = PoissonLFEMSolver(pde, mesh, p, timer=tmr, logger=logger)

s0.gamg_solve(IM)


#!/usr/bin/python3
import numpy as np
from scipy.sparse.linalg import spsolve
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.cfd import NSMacSolver
from tarlor_green_pde import taylor_greenData 

Re = 1
nu = 1/Re

#PDE 模型 
pde = taylor_greenData()
domain = pde.domain()

#空间离散
nx = 4
ny = 4
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh_u = UniformMesh2d([0, nx, 0, ny-1], h=(hx, hy), origin=(domain[0], domain[2]+hy/2))
mesh_v = UniformMesh2d([0, nx-1, 0, ny], h=(hx, hy), origin=(domain[0]+hx/2, domain[2]))
mesh_p = UniformMesh2d([0, nx-1, 0, ny-1], h=(hx, hy), origin=(domain[0]+hx/2, domain[2]+hy/2))


#时间离散
duration = pde.duration()
nt = 128
tau = (duration[1] - duration[0])/nt

solver = NSMacSolver(mesh_u, mesh_v, mesh_p)


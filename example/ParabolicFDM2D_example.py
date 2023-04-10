import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from fealpy.pde.parabolic_2d import SinSinExpPDEData
from fealpy.mesh import UniformMesh2d

# PDE 模型
pde = SinSinExpPDEData()

# 空间离散
domain = pde.domain()
nx = 20
ny = 20
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()

# 时间离散
duration = pde.duration()
nt = 800 
tau = (duration[1] - duration[0])/nt 

uh0 = mesh.interpolate(pde.init_solution, intertype='node')

A = mesh.parabolic_operator_forward(tau)
print("A:", A)

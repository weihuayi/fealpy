import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from fealpy.pde.parabolic_2d import SinSinExpPDEData
from fealpy.mesh import UniformMesh2d

# PDE 模型
pde = SinSinExpPDEData()

# 空间离散
domain = pde.domain()
nx = 40
ny = 40
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()

# 时间离散
duration = pde.duration()
nt = 6400 
tau = (duration[1] - duration[0])/nt 

uh0 = mesh.interpolate(pde.init_solution, intertype='node')
print("uh0", uh0)
print(uh0.shape)


def advance_forward(n):
    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        uh0.flat = A@uh0.flat + tau*f.flat
        gD = lambda p: pde.dirichlet(p, t+tau)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

uh0, t = advance_forward(2)

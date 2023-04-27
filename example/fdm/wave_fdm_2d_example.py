import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import UniformMesh2d
from scipy.sparse.linalg import spsolve
from fealpy.pde.wave_2d import MembraneOscillationPDEData

import ipdb

pde = MembraneOscillationPDEData()

# 格式参数
theta = 0.5

# 空间离散
domain = pde.domain()
nx = 100
ny = 100
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))

# 时间离散
duration = pde.duration()
nt = 1000
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node') # （nx+1, ny+1)
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node') # (nx+1, ny+1)
uh1 = mesh.function('node') # (nx+1, ny+1)

def advance(n, *frags):
    """
    @brief 时间步进

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        rx = tau/hx
        ry = tau/hy 
        uh1[1:-1, 1:-1] = 0.5*rx**2*(uh0[0:-2, 1:-1] + uh0[2:, 1:-1]) + \
                0.5*ry**2*(uh0[1:-1, 0:-2] + uh0[1:-1, 2:]) + \
                (1 - rx**2 - ry**2)*uh0[1:-1, 1:-1] + tau*vh0[1:-1, 1:-1]
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)
        return uh1, t
    else:
        A0, A1, A2 = mesh.wave_operator(tau, theta=theta)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        f += A1@uh1.flat + A2@uh0.flat

        uh0[:] = uh1[:]
        if theta == 0.0:
            uh1.flat = f
            gD = lambda p: pde.dirichlet(p, t+tau)
            mesh.update_dirichlet_bc(gD, uh1)
        else:
            gD = lambda p: pde.dirichlet(p, t+tau)
            A0, f = mesh.apply_dirichlet_bc(gD, A0, f)
            uh1.flat = spsolve(A0, f)

        return uh1, t


box = [0, 1, 0, 1]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance, frames=nt+1)
plt.show()

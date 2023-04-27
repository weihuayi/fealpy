import numpy 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.pde.wave_1d import StringOscillationPDEData

from fealpy.mesh import UniformMesh1d

import ipdb

theta = 0.5

pde = StringOscillationPDEData()
domain = pde.domain()
duration = pde.duration()

nx = 100
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

nt = 1000
tau = (duration[1] - duration[0])/nt

uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')


def advance(n, *frags):
    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        rx = tau/hx 
        uh1[1:-1] = rx**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-rx**2)*uh0[1:-1] + tau*vh0[1:-1]
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)
        return uh1, t
    else:
        A, B, C = mesh.wave_operator(tau, theta=theta)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        f += B@uh1 + C@uh0

        uh0[:] = uh1[:]
        if theta == 0.0:
            uh1[:] = f
            mesh.update_dirichlet_bc(gD, uh1)
        else:
            gD = lambda p: pde.dirichlet(p, t+tau)
            A, f = mesh.apply_dirichlet_bc(gD, A, f)
            uh1[:] = spsolve(A, f)
            
        return uh1, t

box = [0, 1, -0.1, 0.1]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance, frames=nt+1)
plt.show()

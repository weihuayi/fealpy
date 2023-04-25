import numpy 
import matplotlib.pyplot as plt

from fealpy.pde.wave_1d import StringOscillationPDEData

from fealpy.mesh import UniformMesh1d


pde = StringOscillationPDEData()
domain = pde.domain()
duration = pde.duration()

nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

nt = 100
tau = (duration[1] - duration[0])/nt

uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')


A, B, C = mesh.wave_operator(tau, theta=0.5)


def advance(n, *frags):
    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        r = tau/hx**2 
        uh1[1:-1] = r**2*(uh0[0:-2] + uh0[2:])/2.0 + r**2*uh0[1:-1] + tau*vh0[1:-1]
        mesh.update_dirichlet_bc(pde.dirichlet, uh1)
        return uh1, t
    else:
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        uh0[:] = A@uh0 + tau*f
        gD = lambda p: pde.dirichlet(p, t+tau)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

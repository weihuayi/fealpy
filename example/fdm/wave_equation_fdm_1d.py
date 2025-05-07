import ipdb
from matplotlib import pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.pde.wave_1d import StringOscillationPDEData
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
from fealpy.fdm.wave_operator import WaveOperator
from fealpy.solver import spsolve

pde = StringOscillationPDEData()

# 空间离散
domain = pde.domain()
nx = 100
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 1000
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')

# 显格式
def advance_explicit(n, *frags):
    """
    @brief 时间步进格式为显格式

    @param[in] n int, 表示第 n 个时间步 
    """
    a = 1
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
        wave_operator = WaveOperator(mesh)
        A = WaveOperator.assembly(tau, a)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        uh2 = A@uh1 - uh0 + f  # 根据差分方程进行更新
        gD = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh2)
        uh0[:] = uh1
        uh1[:] = uh2
        return uh1, t


'''
box = [0, 1, -0.1, 0.1]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_explicit, fname='explicit.mp4', frames=nt+1)
plt.show()
'''

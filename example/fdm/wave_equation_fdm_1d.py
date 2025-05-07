import ipdb
from matplotlib import pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.pde.wave_1d import StringOscillationPDEData
from fealpy.mesh import UniformMesh
from fealpy.fdm.wave_operator import WaveOperator
from fealpy.fdm.dirichlet_bc import DirichletBC
from fealpy.solver import spsolve
from functools import partial

pde = StringOscillationPDEData()

# 空间离散
domain = pde.domain()
nx = 400
hx = (domain[1] - domain[0])/nx
extent = [0, nx]
mesh = UniformMesh(domain, extent)

# 时间离散
duration = pde.duration()
nt = 500
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node').reshape(-1,1)

# 显格式
def advance_explicit(n, *frags):
    """
    @brief 时间步进格式为显格式

    @param[in] n int, 表示第 n 个时间步 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        rx = tau/hx
        uh1[1:-1] = rx**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-rx**2)*uh0[1:-1] + tau*vh0[1:-1]
        gD = lambda p: pde.dirichlet(p, t)
        return uh1, t
    else:
        wave_operator = WaveOperator(mesh)
        A = wave_operator.assembly(tau=tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source)
        f *= tau**2
        uh2 = A@uh1 - uh0 + f  # 根据差分方程进行更新
        gD = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh2)
        uh0[:] = uh1
        uh1[:] = uh2
        
        return uh1, t
'''
box = [0, 1, -2, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_explicit, fname='explicit.mp4', frames=nt+1)
plt.show()
'''

#隐格式
def advance_implicit(n, theta=0.25, *frags):
    """
    @brief 时间步进格式为隐格式

    @param[in] n int, 表示第 n 个时间步（当前时间步） 
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
        wave_operator = WaveOperator(mesh)
        A0, (A1, A2) = wave_operator.implicit_assembly(tau=tau, theta=theta)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source)
        f *= tau**2
        f += A1@uh1 + A2@uh0

        uh0[:] = uh1[:]
        gD = lambda p: pde.dirichlet(p, t + tau)

        dirichlet_bc = DirichletBC(mesh, gD)
        A0, f = dirichlet_bc.apply(A0, f)
        uh1[:] = spsolve(A0, f)
            
        return uh1, t


advance_implicit_theta = partial(advance_implicit, theta=0.25)

box = [0, 1, -2, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_implicit_theta, fname='implicit.mp4', frames=nt+1)
plt.show()

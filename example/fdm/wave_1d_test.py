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

# 空间离散
domain = pde.domain()
nx = 400
hx = (domain[1] - domain[0])/nx
extent = [0, nx]
mesh = UniformMesh(domain, extent)

# 时间离散
duration = pde.duration()
nt_init = 500
tau_init = (duration[1] - duration[0])/nt_init
fixed_time = 0.5

maxit = 5
em = bm.zeros((3, maxit), dtype=bm.float64)
h_sizes = bm.zeros(maxit, dtype=bm.float64)

for i in range(maxit):
    hx = (domain[1] - domain[0])/mesh.nx
    h_sizes[i] = hx

    tau = tau_init*(h_sizes[i]/h_sizes[0])

    nt = int((duration[1] - duration[0]) / tau)
    fixed_time_step = int((fixed_time - duration[0]) / tau)
    
    # 准备初值
    uh0 = mesh.interpolate(pde.init_solution, 'node').flatten()
    vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node').flatten()
    uh1 = mesh.function('node').flatten()

    for n in range(fixed_time_step + 1):
        uh, t = advance_explicit(n)

    solution = lambda p: pde.solution(p, fixed_time)
    em[0, i], em[1, i], em[2, i] = mesh.error(solution, uh1)
    print(em)
    if i < maxit:
        mesh.uniform_refine()

print("em_ratio:\n", em[:, 0:-1]/em[:, 1:])

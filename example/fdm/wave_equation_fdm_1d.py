import ipdb
from matplotlib import pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')
from fealpy.pde.wave_1d import StringOscillationPDEData
from fealpy.mesh import UniformMesh
from fealpy.fdm.wave_operator import WaveOperator
from fealpy.fdm.dirichlet_bc import DirichletBC
from fealpy.solver import spsolve
from functools import partial


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

# 固定某一时间离散
nt_init = 500
tau_init = (duration[1] - duration[0])/nt_init
fixed_time = 0.5

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node').flatten()
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node').flatten()
uh1 = mesh.function('node').flatten()


def wave_1d_run(nx, nt, time_step, error=True, show=True):
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
            # uh, t = advance_explicit(n)
            uh, t = advance_implicit(n)

        solution = lambda p: pde.solution(p, fixed_time)
        em[0, i], em[1, i], em[2, i] = mesh.error(solution, uh1)
        if i < maxit:
            mesh.uniform_refine()

    print("em:\n", em)
    print("em_ratio:\n", em[:, 0:-1]/em[:, 1:])
    
    box = [0, 1, -1.5, 1.5]
    fig, axes = plt.subplots()
    
    # 显格式可视化
    mesh.show_animation(fig, axes, box, advance_explicit, fname='explicit.mp4', frames=nt+1)
    
    # 隐格式可视化
    advance_implicit_theta = partial(advance_implicit, theta=0.25)
    mesh.show_animation(fig, axes, box, advance_implicit_theta, fname='implicit.mp4', frames=nt+1)

    plt.show()

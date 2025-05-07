from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh
from fealpy.solver import spsolve
from fealpy.pde.wave_2d  import MembraneOscillationPDEData
from fealpy.fdm.wave_operator import WaveOperator
from fealpy.fdm.dirichlet_bc import DirichletBC

pde = MembraneOscillationPDEData()

def advance_explicit(n, *frags):
    """时间步进为显格式

    Parameters:
    - n: 表示第 n 个时间步
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        rx = tau/hx
        ry = tau/hy
        nx, ny = mesh.nx, mesh.ny  
        uh0_2d = uh0.reshape((ny+1, nx+1)) 
        vh0_2d = vh0.reshape((ny+1, nx+1))

        uh1_2d = uh1.reshape((ny+1, nx+1))
        uh1_2d[1:-1, 1:-1] = 0.5*rx**2*(uh0_2d[0:-2, 1:-1] + uh0_2d[2:, 1:-1]) + \
                0.5*ry**2*(uh0_2d[1:-1, 0:-2] + uh0_2d[1:-1, 2:]) + \
                (1 - rx**2 - ry**2)*uh0_2d[1:-1, 1:-1] + tau*vh0_2d[1:-1, 1:-1]
        uh1[:] = uh1_2d.flatten()

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)
        return uh1, t
    else:
        wave_operator = WaveOperator(mesh)
        A = wave_operator.assembly(tau=tau)
        uh2 = A @ uh1 - uh0
        gD = lambda p: pde.dirichlet(p, t + tau)

        mesh.update_dirichlet_bc(gD, uh2)

        uh0[:] = uh1
        uh1[:] = uh2

        return uh1, t
'''
# 空间离散
domain = pde.domain()
nx = 10
ny = 10
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
extent = [0, nx, 0, ny]
mesh = UniformMesh(domain, extent)

# 时间离散
duration = pde.duration()
nt_init = 100
tau_init = (duration[1] - duration[0]) / nt_init
fixed_time = 0.7

maxit = 5
em = bm.zeros((3, maxit), dtype=bm.float64)
h_sizes = bm.zeros(maxit, dtype=bm.float64)
for i in range(maxit):
    hx = (domain[1] - domain[0]) / mesh.nx
    hy = (domain[3] - domain[2]) / mesh.ny
    h_sizes[i] = max(hx, hy)

    tau = tau_init * (h_sizes[i] / h_sizes[0])

    nt = int((duration[1] - duration[0]) / tau)
    fixed_time_step = int((fixed_time - duration[0]) / tau)

    uh0 = mesh.interpolate(pde.init_solution, 'node')
    vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
    uh1 = mesh.function('node').flatten()

    for n in range(fixed_time_step + 1):
        uh, t = advance_explicit(n)

    solution = lambda p: pde.solution(p, fixed_time)
    em[0, i], em[1, i], em[2, i] = mesh.error(solution, uh1)

    if i < maxit:
        mesh.uniform_refine()

print("em_ratio:\n", em[:, 0:-1]/em[:, 1:])
'''
'''
# 显格式二维画图
box = [0, 1, 0, 1, -1, 1]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_explicit, 
                    fname='explicit.mp4', plot_type='imshow', frames=nt+1)
plt.show()

# 显格式三维画图
box = [0, 1, 0, 1, -1, 1]
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.show_animation(fig, axes, box, advance_explicit, 
                    fname='explicit.mp4', plot_type='surface', frames=nt+1)
plt.show()
'''

def advance_implicit(n, *frags):
    """
    @brief 时间步进为隐格式

    @param[in] n int, 表示第 n 个时间步
    """
    global uh1
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        rx = tau/hx
        ry = tau/hy
        nx, ny = mesh.nx, mesh.ny  
        uh0_2d = uh0.reshape((ny+1, nx+1)) 
        vh0_2d = vh0.reshape((ny+1, nx+1))

        uh1_2d = uh1.reshape((ny+1, nx+1))
        uh1_2d[1:-1, 1:-1] = 0.5*rx**2*(uh0_2d[0:-2, 1:-1] + uh0_2d[2:, 1:-1]) + \
                0.5*ry**2*(uh0_2d[1:-1, 0:-2] + uh0_2d[1:-1, 2:]) + \
                (1 - rx**2 - ry**2)*uh0_2d[1:-1, 1:-1] + tau*vh0_2d[1:-1, 1:-1]
        uh1[:] = uh1_2d.flatten()

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)
        return uh1, t
    else:
        wave_operator = WaveOperator(mesh)
        A0, (A1, A2) = wave_operator.implicit_assembly(tau=tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source)
        f *= tau**2
        f.flat += A1@uh1 + A2@uh0

        uh0[:] = uh1[:]
        gD = lambda p: pde.dirichlet(p, t + tau)

        dirichlet_bc = DirichletBC(mesh, gD)
        A0, f = dirichlet_bc.apply(A0, f)
        uh1 = spsolve(A0, f, solver='scipy')

        return uh1, t

'''
# 空间离散
domain = pde.domain()
nx = 10
ny = 10
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
extent = [0, nx, 0, ny]
mesh = UniformMesh(domain, extent)

# 时间离散
duration = pde.duration()
nt_init = 100
tau_init = (duration[1] - duration[0]) / nt_init
fixed_time = 0.7

maxit = 4
em = bm.zeros((3, maxit), dtype=bm.float64)
h_sizes = bm.zeros(maxit, dtype=bm.float64)
for i in range(maxit):
    hx = (domain[1] - domain[0]) / mesh.nx
    hy = (domain[3] - domain[2]) / mesh.ny
    h_sizes[i] = max(hx, hy)

    tau = tau_init * (h_sizes[i] / h_sizes[0])

    nt = int((duration[1] - duration[0]) / tau)
    fixed_time_step = int((fixed_time - duration[0]) / tau)

    uh0 = mesh.interpolate(pde.init_solution, 'node')
    vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
    uh1 = mesh.function('node').flatten()

    for n in range(fixed_time_step + 1):
        uh, t = advance_implicit(n)

    solution = lambda p: pde.solution(p, fixed_time)
    em[0, i], em[1, i], em[2, i] = mesh.error(solution, uh1)

    if i < maxit:
        mesh.uniform_refine()

print("em_ratio:\n", em[:, 0:-1]/em[:, 1:])
'''
'''
# 隐格式二维动画
box = [0, 1, 0, 1, -1, 1]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_implicit,
                    fname='implicit.mp4', plot_type='imshow', frames=nt+1)
plt.show()

# 隐格式三维动画
box = [0, 1, 0, 1, -1, 1]

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.show_animation(fig, axes, box, advance_implicit,
                    fname='implicit.mp4', plot_type='surface', frames=nt+1)
plt.show()
'''
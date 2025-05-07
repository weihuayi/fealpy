from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh2d
from scipy.sparse.linalg import spsolve
from fealpy.pde.wave_2d  import MembraneOscillationPDEData
from fealpy.fdm.wave_operator import WaveOperator
from fealpy.fdm.dirichlet_bc import DirichletBC

pde = MembraneOscillationPDEData()

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
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node').flatten()

def advance_explicit(n, *frags):
    """
    @brief 时间步进为显格式

    @param[in] n int, 表示第 n 个时间步
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
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source)
        f *= tau**2
        uh2 = A @ uh1 - uh0
        gD = lambda p: pde.dirichlet(p, t + tau)

        mesh.update_dirichlet_bc(gD, uh2)

        uh0[:] = uh1
        uh1[:] = uh2

        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh1.flatten(), errortype='l2')
        #print(f"the error is {e}")

        return uh1, t

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
        #A0, f = mesh.apply_dirichlet_bc(gD, A0, f)

        dirichlet_bc = DirichletBC(mesh, gD)
        A0, f = dirichlet_bc.apply(A0, f, uh0)
        print("A0:", A0.shape)
        uh1 = spsolve(A0, f)

        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh1, errortype='max')
        print(f"the max error is {e}")

        return uh1, t

box = [0, 1, 0, 1, -1, 1]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_implicit,
                    fname='implicit.mp4', plot_type='imshow', frames=nt+1)
plt.show()


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
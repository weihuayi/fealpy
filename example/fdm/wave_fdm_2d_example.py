import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from fealpy.mesh import UniformMesh2d
from scipy.sparse.linalg import spsolve
from fealpy.pde.wave_2d import MembraneOscillationPDEData


# 参数解析
parser = argparse.ArgumentParser(description=
        """
        二维均匀网格上波动方程的有限差分方法，
        边界条件为的带纯 Dirichlet 型，
        已知初始时刻解函数及其关于时间的偏导函数的表达式
        """)

parser.add_argument('--nx',
        default=1000, type=int,
        help="x 方向上的剖分段数，默认为 1000 段.")

parser.add_argument('--ny',
        default=1000, type=int,
        help="y 方向上的剖分段数，默认为 1000 段.")

parser.add_argument('--nt',
        default=4000, type=int,
        help='时间剖分段数，默认为 4000 段.')


parser.add_argument('--theta',
        default=0.0, type=float,
        help='离散格式参数，默认 0.0.')

args = parser.parse_args()

nx = args.nx
ny = args.ny
nt = args.nt
theta = args.theta

pde = MembraneOscillationPDEData()

# 空间离散
domain = pde.domain()
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))

# 时间离散
duration = pde.duration()
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node') # （nx+1, ny+1)
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node') # (nx+1, ny+1)
uh1 = mesh.function('node') # (nx+1, ny+1)

A = mesh.wave_operator_explicity(tau)

def advance_explicity(n, *frags):
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
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        f += A@uh1.flat - uh0.flat

        uh0[:] = uh1[:]
        uh1.flat = f

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)

        return uh1, t


box = [0, 1, 0, 1, -0.021, 0.021]
fig = plt.figure()

#axes = fig.add_subplot(111, projection='3d')
#mesh.show_animation(fig, axes, box, advance_explicity, plot_type='surface', frames=nt+1)
axes = fig.add_subplot()
mesh.show_animation(fig, axes, box, advance_explicity, frames=nt+1, plot_type='imshow')
plt.show()

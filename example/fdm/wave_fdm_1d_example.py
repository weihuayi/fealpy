import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from typing import Callable, Tuple, Any
from fealpy.pde.wave_1d import StringOscillationPDEData
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        一维均匀网格上用有限差分方法求解波动方程，其中
        边界条件为纯 Dirichlet 边界条件
        已知初始时刻解函数及其关于时间的偏导函数的表达式
        """)

parser.add_argument('--nx',
        default=100, type=int,
        help='空间剖分段数，默认为 100 段.')

parser.add_argument('--nt',
        default=1000, type=int,
        help='时间剖分段数，默认为 1000 段.')

parser.add_argument('--theta',
        default=0.5, type=float,
        help='离散格式参数，默认 0.5.')

args = parser.parse_args()

theta = args.theta 
nx = args.nx
nt = args.nt

pde = StringOscillationPDEData()

# 空间离散
domain = pde.domain()
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
tau = (duration[1] - duration[0])/nt

# 初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')

# 定义时间步进函数
def advance(n: int, *frags: Any) -> Tuple[np.float64, float]:
    """
    @brief 波动方程的时间步进程序 

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
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        f += B@uh1 + C@uh0

        uh0[:] = uh1
        gD = lambda p: pde.dirichlet(p, t)
        if theta == 0.0:
            uh1[:] = f
            mesh.update_dirichlet_bc(gD, uh1)
        else:
            A, f = mesh.apply_dirichlet_bc(gD, A, f)
            uh1.flat = spsolve(A, f)
            
        return uh1, t

box = [0, 1, -0.1, 0.1]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance, frames=nt+1)
plt.show()

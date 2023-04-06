import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

import ipdb # 程序调试工具

from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh1d

import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.parabolic_1d import SinExpPDEData
from fealpy.mesh import UniformMesh1d
from fealpy.timeintegratoralg import UniformTimeLine


# PDE 模型
pde = SinExpPDEData()

# 空间离散
domain = pde.domain()
nx = 100 
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()

# 时间离散
duration = pde.duration()
nt = 1000 
timeline = UniformTimeLine(duration[0], duration[1], nt)
tau = timeline.dt

uh0 = mesh.interpolate(pde.init_solution, intertype='node')


def advance_forward(n):
    """
    @brief 时间步进格式为向前欧拉方法
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        uh0[:] = A@uh0 + tau*f
        uh0[isBdNode] = pde.dirichlet(node[isBdNode], t + tau)
        
        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
    
def advance_backward(n):
    """
    @brief 时间步进格式为向后欧拉方法
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_backward(tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += uh0
        gD = lambda p: pde.dirichlet(p, t+tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)

        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t

def advance_crank_nicholson(n):
    """
    @brief 时间步进格式为 CN 方法  
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += B@uh0
        gD = lambda p: pde.dirichlet(p, t+tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)
        return uh0, t



fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_backward, frames=nt + 1)
plt.show()

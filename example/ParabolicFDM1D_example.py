import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from fealpy.pde.parabolic_1d import SinExpPDEData
from fealpy.mesh import UniformMesh1d

# PDE 模型
pde = SinExpPDEData()

# 空间离散
domain = pde.domain()
nx = 40 
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()

# 时间离散
duration = pde.duration()
nt = 3200 
tau = (duration[1] - duration[0])/nt 
print("网比r:", tau/(hx**2))
uh0 = mesh.interpolate(pde.init_solution, intertype='node')

def advance_forward(n, *frags):
    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        uh0[:] = A@uh0 + tau*f
        gD = lambda p: pde.dirichlet(p, t+tau)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
    
def advance_backward(n):
    """
    @brief 时间步进格式为向后欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
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

        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t


fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5
def init_callback():
    # 调用 advance(0) 获取初始解
    uh0, _ = advance_forward(0)
    return uh0

# mesh.show_animation(fig, axes, box, advance_forward, init=init_callback, frames=nt + 1)
mesh.show_animation(fig, axes, box, advance_crank_nicholson, init=init_callback, frames=nt + 1)
plt.show()

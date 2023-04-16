import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from fealpy.pde.parabolic_2d import SinSinExpPDEData
from fealpy.mesh import UniformMesh2d

# PDE 模型
pde = SinSinExpPDEData()

# 空间离散
domain = pde.domain()
nx = 40
ny = 40
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()

# 时间离散
duration = pde.duration()
nt = 6400 
tau = (duration[1] - duration[0])/nt 
print("网比r_x:", tau/(hx**2)) # 0.25
print("网比r_y:", tau/(hy**2)) # 0.25

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node') # uh0.shape = (nx+1, ny+1)
print("uh0:", uh0.shape)

def advance_forward(n, *fargs):
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
        
        uh0[:].flat = A@uh0[:].flat + (tau*f[:]).flat
        gD = lambda p: pde.dirichlet(p, t+tau)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t


def advance_backward(n, *fargs):
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
        uh0.flat[:] = spsolve(A, f)
        
        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

def advance_crank_nicholson(n, *fargs): 
    """
    @brief 时间步进格式为 CN 方法  
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node') # f.shape = (nx+1,ny+1)
        f *= tau
        f.flat[:] += B@uh0.flat[:]
         
        gD = lambda p: pde.dirichlet(p, t+tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0.flat[:] = spsolve(A, f)

        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t


fig, axes = plt.subplots()
box = [0, 1, 0, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1
# mesh.show_animation(fig, axes, box, advance_forward, frames=nt + 1)
# mesh.show_animation(fig, axes, box, advance_backward, frames=nt + 1)
mesh.show_animation(fig, axes, box, advance_crank_nicholson, frames=nt + 1)
plt.show()

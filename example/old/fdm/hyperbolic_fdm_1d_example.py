import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from fealpy.pde.hyperbolic_1d import Hyperbolic1dPDEData
from fealpy.mesh import UniformMesh1d

pde = Hyperbolic1dPDEData()

domain = pde.domain()
nx = 40
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

duration = pde.duration()
nt = 3200
tau = (duration[1] - duration[0])/nt

uh0 = mesh.interpolate(pde.init_solution, intertype='node')

def hyperbolic_explicity_windward(n, *fargs):
    """
    @brief 时间步进格式为迎风格式 

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_explicity_upwind(tau)
        uh0[:] = A@uh0

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

def hyperbolic_lax(n, *fargs): 
    """
    @brief 时间步进格式为守恒型 Lax 格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_explicity_lax_friedrichs(pde.a(), tau)
        uh0[:] = A@uh0
        
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

def hyperbolic_windward(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为中心差分格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_central(1, tau)
        uh0[:] = A@uh0
        
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)
        
        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

def hyperbolic_windward_with_vicious(n, *fargs): 
    """
    @brief 时间步进格式为带粘性项的显式迎风格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_explicity_upwind_viscous(pde.a(), tau)
        uh0[:] = A@uh0
        
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)
        
        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic_windward, frames=nt+1)
plt.show()

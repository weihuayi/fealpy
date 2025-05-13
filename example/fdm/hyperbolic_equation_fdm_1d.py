import matplotlib.pyplot as plt
from typing import List
from fealpy.backend import backend_manager as bm
from fealpy.fdm import  DirichletBC
from fealpy.fdm.hyperbolic_operator import HyperbolicOperator
from fealpy.pde.hyperbolic_1d import Hyperbolic1dPDEData
from fealpy.mesh import UniformMesh
from fealpy.typing import TensorLike
from fealpy.solver import spsolve
bm.set_backend('numpy')
# PDE 模型
pde = Hyperbolic1dPDEData()
domain = [0,1]
nx = 10
extent = [0,nx]
mesh = UniformMesh(domain, extent)
# 时间离散
duration = pde.duration()
nt = 600
tau = (duration[1] - duration[0]) / nt 
a = pde.a()
uh0 = mesh.interpolate(pde.init_solution)
P0 = HyperbolicOperator(mesh,tau,a)
def hyperbolic(n): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为迎风格式,守恒型 Lax 格式,中心差分格式,带粘性项的显式迎风格式(四者是相同的)
    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = P0.assembly()
        uh0[:] = A@uh0
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)
        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
box = [0, 1, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic, frames=nt+1)
plt.show()  
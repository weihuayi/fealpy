from fealpy.backend import backend_manager as bm
from fealpy.fdm import HyperbolicOperator
from fealpy.mesh import UniformMesh
from fealpy.fdm import DirichletBC
from fealpy.solver import spsolve
import matplotlib.pyplot as plt
from fealpy.pde.hyperbolic_2d import Hyperbolic2dPDEData
from fealpy.pde.hyperbolic_2d_sympy import Hyperbolic2dData

domain = [0, 1, 0, 1]
extent = [0, 20, 0, 20]
mesh = UniformMesh(domain,extent)
duration = [0,1]
#pde = Hyperbolic2dPDEData()
pde = Hyperbolic2dData("sin(pi*x)*sin(pi*y)*cos(pi*t)", a=1.0)
nt =400
tau = (duration[1] - duration[0]) / nt

uh0 = mesh.interpolate(pde.init_solution)
a = pde.a
H0 = HyperbolicOperator(mesh,tau,a, method='explicity_upwind_viscous')
em = bm.zeros((3, 1), dtype=bm.float64)

def hyperbolic_windward(n):
    """
    @brief 时间步进格式为迎风格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = H0.explicity_upwind_viscous_assembly()
        source  = lambda p: pde.source(p, t+tau)
        f = mesh.interpolate(source)
        uh0[:]=A@uh0 + tau*f
        gD = lambda p: pde.dirichlet(p, t+tau)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution = lambda p: pde.solution(p, t + tau)

        em[0, 0], em[1, 0], em[2, 0] = mesh.error(solution, uh0)
        print(em[0, 0], em[1, 0], em[2, 0])
        return uh0, t


fig = plt.figure()
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
axes = fig.add_subplot(111, projection='3d')
mesh.show_animation(fig, axes, box, hyperbolic_windward, 
                    fname='wind.mp4', plot_type='surface', frames=nt + 1)
plt.show()

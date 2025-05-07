      
from fealpy.backend import backend_manager as bm
from fealpy.fdm import ParabolicOperator
from fealpy.pde.parabolic2d import Parabolic2dData
from fealpy.sparse import csr_matrix,spdiags
from fealpy.mesh import UniformMesh
from fealpy.fdm import DirichletBC
from fealpy.solver import spsolve
import matplotlib.pyplot as plt

domain = [0,1,0,1]
extent = [0, 10, 0, 10]
n = extent[1]
h = domain[1]-domain[0]/n
mesh = UniformMesh(domain, extent)

duration = [0,1]

pde = Parabolic2dData('exp(-20*t)*sin(4*pi*x)*sin(4*pi*y)', 'x', 'y', 't', D=domain, T=duration)

nt = 800
tau = (duration[1]-duration[0])/nt

uh0 = mesh.interpolate(pde.init_solution)

P0 = ParabolicOperator(mesh,tau,method='backward')
em = bm.zeros((3, 1), dtype=bm.float64)
def advance_backward(n):
    """
    @brief 时间步进格式为向后欧拉方法
    
    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    bc = DirichletBC(mesh, lambda p: pde.dirichlet(p, t+tau))
    if n == 0:
        return uh0, t
    else:
        A = P0.backward_assembly()
        
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source)
        f *= tau
        f += uh0

        A, f = bc.apply(A, f)
        uh0.flat = spsolve(A, f,solver='scipy')
            
        solution = lambda p: pde.solution(mesh.node, t + tau)
        # em[0, 0], em[1, 0], em[2, 0] = mesh.error(pde.solution, uh0)
        return uh0, t

# fig, axes = plt.subplots()
# box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
# mesh.show_animation(fig, axes, box, advance_backward, fname='parabolic_af.mp4', plot_type='imshow', frames=nt + 1)
# plt.show()

fig = plt.figure()
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
axes = fig.add_subplot(111, projection='3d')
mesh.show_animation(fig, axes, box, advance_backward, 
                    fname='parabolic_ab.mp4', plot_type='surface', frames=nt + 1)
plt.show()


    
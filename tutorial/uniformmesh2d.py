import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import UniformMesh1d
# 定义一个 PDE 的模型类
class PDEModel:
    def domain(self):
        return [0, 1]

    def solution(self, x):
        return np.sin(4*np.pi*x)

    def gradient(self, x):
        return 4*np.pi*np.cos(4*np.pi*x)
        
    def source(self, x):
        return 16*np.pi**2*np.sin(4*np.pi*x)

pde = PDEModel()
domain = pde.domain()


# [0, 1] 区间均匀剖分 5 段，每段长度 0.2
nx = 5
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

uI = mesh.interpolation(pde.solution, 'node')
A = mesh.laplace_operator_with_dbc()
F = mesh.interpolation(pde.source, 'node')
F[0] = uI[0]
F[-1] = uI[-1]
uh = spsolve(A, F)

fig = plt.figure()
axes = fig.gca()
mesh.show_function(axes, uh)

# 画出网格
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
plt.show()

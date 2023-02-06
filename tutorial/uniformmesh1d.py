import numpy as np 
import matplotlib.pyplot as plt

from fealpy.mesh import UniformMesh1d

class PDEModel:
    def __init__(self):
        pass

    def domain(self):
        return [0, 1]

    def solution(self, x):
        return np.sin(4*np.pi*x)

    def gradient(self, x):
        return 4*np.pi*np.cos(4*np.pi*x)


pde = PDEModel()
domain = pde.domain()

# [0, 1] 区间均匀剖分 10 段，每段长度 0.1 
nx = 5 
h = 1/nx
mesh = UniformMesh1d([0, nx], h=h, origin=0.0)

A = mesh.laplace_operator_with_dbc()
print(A.toarray())

A = mesh.laplace_operator()
print(A.toarray())


uh0 = mesh.interpolation(pde.solution, 'node')
uh1 = mesh.interpolation(pde.solution, 'cell')


fig = plt.figure()
axes = fig.gca()
mesh.show_function(axes, uh0)

fig = plt.figure()
axes = fig.gca()
x = np.linspace(domain[0], domain[1], 100)
uh = pde.solution(x)
axes.plot(x, uh)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

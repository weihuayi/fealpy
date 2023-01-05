import numpy as np 
import matplotlib.pyplot as plt

from fealpy.mesh import UniformMesh1d

def u(x):
    return np.sin(2*np.pi*x)

# [0, 1] 区间均匀剖分 10 段，每段长度 0.1 
nx = 10
h = 1/nx
mesh = UniformMesh1d([0, nx], h=h, origin=0.0)

uh0 = mesh.interpolation(u, 'node')
uh1 = mesh.interpolation(u, 'cell')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

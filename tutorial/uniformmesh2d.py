import numpy as np 
import matplotlib.pyplot as plt

from fealpy.mesh import UniformMesh2d

def u(p):
    x = p[..., 0]
    y = p[..., 1]
    return np.cos(np.pi*x)*np.cos(np.pi*y)

mesh = UniformMesh2d([0, 5, 0, 5], h=(0.2, 0.2), origin=(0.0, 0.0))


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

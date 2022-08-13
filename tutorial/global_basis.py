
import numpy as np
import matplotlib.pyplot as plt


from fealpy.mesh import MeshFactory as MF


def f(p):
    x = p[..., 0] # x.shape == (NQ, NC)
    y = p[..., 1] # y.shape == (NQ, NC)
    return np.exp(x**2 + y**2) # (NQ, NC)




domain = [0, 1, 0, 1]

mesh = MF.boxmesh2d(domain, nx=5, ny=5, meshtype='tri')

node = mesh.entity('node')
cell = mesh.entity('cell')
NN = mesh.number_of_nodes()

uh = np.zeros(NN, dtype=np.float64)
uh[20] = 1.0 # \phi_20


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=30)

fig = plt.figure()
axes= fig.add_subplot(projection='3d')
axes.plot_trisurf(node[:, 0], node[:, 1], cell, uh, cmap='rainbow', lw=3, edgecolors='k')
plt.show()

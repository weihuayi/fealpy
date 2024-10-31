import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.pde.Stokes_Model_2d import SinSinData
from fealpy.pde.Stokes_Model_2d import PolyData
from fealpy.pde.Stokes_Model_2d import CosSinData


box = [0, 1, 0 , 1]
alpha = 1
nu = 1

#pde = SinSinData(box, alpha, nu)
pde = CosSinData(box)
#pde = PolyData(box)
tmesh = pde.init_mesh(n=5, meshtype='tri')
qmesh = pde.init_mesh(n=2, meshtype='quad')
pmesh = pde.init_mesh(n=2, meshtype='Poly')
#fig = plt.figure()
#axes = fig.gca()
#tmesh.add_plot(axes)
#plt.show()
#fig = plt.figure()
#axes = fig.gca()
#qmesh.add_plot(axes)
#plt.show()
#fig = plt.figure()
#axes = fig.gca()
#pmesh.add_plot(axes)
#plt.show()
#
node = qmesh.entity('node')
idx = np.shape(node[:, 0])
n = int(np.sqrt(idx))
xx = node[:, 0].reshape(n, n)
yy = node[:, 1].reshape(n, n)
p = pde.pressure(node)
u = pde.velocity(node)
u1 = u[:, 0]
figure = plt.figure()
axes = Axes3D(figure)
axes.plot_surface(xx.T, yy.T, u1.reshape(n, n).T)
plt.show()

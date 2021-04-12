
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from fealpy.pde.navier_stokes_equation_2d import SinCosData
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace

pde = SinCosData()
mesh = MF.boxmesh2d(pde.box, nx=10, ny=10, meshtype='tri')

uspace = LagrangeFiniteElementSpace(mesh, p=2)
pspace = LagrangeFiniteElementSpace(mesh, p=1)

pI = pspace.interpolation(pde.pressure)
uI = uspace.interpolation(pde.velocity)

#  [[phi, 0], [0, phi]] 
#  [[phi_x, phi_y], [0, 0]] [[0, 0], [phi_x, phi_y]]

A = uspace.stiff_matrix()
B = uspace.div_matrix(pspace)





fig = plt.figure()
axes = fig.gca()
bc = np.array([1/3]*3)
point = mesh.bc_to_point(bc)
p = pI.value(bc)
u = uI.value(bc)
mesh.add_plot(axes, cellcolor=p)
axes.quiver(point[:, 0], point[:, 1], u[:, 0], u[:, 1])
#TODO: add color bar

fig = plt.figure()
axes = fig.gca(projection='3d')
pI.add_plot(axes, cmap='rainbow')
plt.show()




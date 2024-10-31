import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace 
from fealpy.quadrature.TriangleQuadrature import TriangleQuadrature 
from fealpy.mesh.simple_mesh_generator import unitsquaredomainmesh
from fealpy.mesh import meshio 



def u(p):
    x=p[..., 0]
    y=p[..., 1]
    z = np.exp(x+y)
    return z


qf = TriangleQuadrature(3)
bcs, ws = qf.quadpts, qf.weights 

mesh = unitsquaredomainmesh(0.1)
rho = u(mesh.point)
rho /= np.max(rho)



meshio.write_obj_mesh(mesh, 'test.obj')
np.savetxt('rho.txt', rho)


fig1 = plt.figure()
fig1.set_facecolor('white')
axes = fig1.gca() 
mesh.add_plot(axes, cellcolor='w')
plt.show()


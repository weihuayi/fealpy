import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh 
from fealpy.functionspace import LagrangeFESpace

def dis(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.zeros(len(x), dtype=np.float64)
    val[np.abs(y-0.5)<1e-5] = 1
    return val
p = 2
mesh = TriangleMesh.from_unit_square()
space = LagrangeFESpace(mesh, p=p)

#fig = plt.figure()
#axes = fig.add_subplot(111)
#mesh.add_plot(axes)
#plt.show()

ipoint = mesh.interpolation_points(p=p)
u = space.function()
u[:] = dis(ipoint)

mesh.nodedata['uh'] = u
fname = 'test0.vtu'
mesh.to_vtk(fname=fname)

cell2dof = mesh.cell_to_ipoint(p=p)

isMarkedCell = np.abs(np.sum(u[cell2dof], axis=-1))>1.5
data = {'uh':u[cell2dof]}
option = mesh.bisect_options(disp=False, data=data)
mesh.bisect(isMarkedCell, options=option)

space = LagrangeFESpace(mesh, p=p)
cell2dof = space.cell_to_dof()
u = space.function()

u[cell2dof] = option['data']['uh']

mesh.nodedata['uh'] = u
fname = 'test1.vtu'
mesh.to_vtk(fname=fname)


fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
plt.show()
 

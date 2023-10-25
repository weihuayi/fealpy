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
mesh = TriangleMesh.from_unit_square()
space = LagrangeFESpace(mesh, p=2)

#fig = plt.figure()
#axes = fig.add_subplot(111)
#mesh.add_plot(axes)
#plt.show()

node = mesh.entity('node')
u = space.function()
u[:] = dis(node)
NC = mesh.number_of_cells()
H = np.zeros(NC, dtype=np.float64)

cell = mesh.entity(etype=2)
H = np.sum(u[:][cell], axis=-1)

mesh.nodedata['uh'] = u
mesh.celldata['h'] = H
fname = 'test0.vtu'
mesh.to_vtk(fname=fname)

cell2dof = mesh.cell_to_ipoint(p=1)

isMarkedCell = np.abs(np.sum(u[cell2dof], axis=-1))>1.5
data = {'uh':u, 'H':H}
option = mesh.bisect_options(disp=False, data=data)
mesh.bisect(isMarkedCell, options=option)

space = LagrangeFESpace(mesh, p=1)
cell2dof = space.cell_to_dof()
u = space.function()
NC = mesh.number_of_cells()
H = np.zeros(NC, dtype=np.float64)

u = option['data']['uh']
H = option['data']['H']

mesh.nodedata['uh'] = u
mesh.celldata['h'] = H
fname = 'test1.vtu'
mesh.to_vtk(fname=fname)


fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
plt.show()


import numpy as np
import gmsh
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from TriRadiusRatio import TriRadiusRatio

mesh = TriangleMesh.from_meshio('area_EPI_adaptive.vtu')
node = mesh.entity('node')
cell = mesh.entity('cell')
node = node[:,:2]
mesh = TriangleMesh(node, cell)

angle = mesh.angle()
max_angle = np.max(angle,axis=1)
angles = max_angle*(180/np.pi)
m90 = np.sum(angles>90)

fig,axes1= plt.subplots()
axes1.set_ylim(0,8500)
fig.text(0.15,0.85,f'Number of angles>90:{m90}',fontsize=12)
mesh.show_angle(axes1,max_angle)
plt.show()

mesh.celldata['angles'] = angles
mesh.to_vtk(fname='beforeopt_EPI.vtu')
mesh.delete_degree_4()
isBdNode = mesh.ds.boundary_node_flag()
isFreeNode = ~isBdNode
# 去除度为4的点后进行优化
opt = TriRadiusRatio(mesh)
opt.iterate_solver(method='Bjacobi',isFreeNode=isFreeNode)
angle = mesh.angle()
max_angle = np.max(angle,axis=1)
angles = max_angle*(180/np.pi)
m90 = np.sum(angles>90)

fig,axes1= plt.subplots()
axes1.set_ylim(0,8500)
fig.text(0.15,0.85,f'Number of angles>90:{m90}',fontsize=12)
mesh.show_angle(axes1,max_angle)
plt.show()

mesh.celldata['angles'] = angles
mesh.to_vtk(fname='optEPI.vtu')
gmsh.finalize()


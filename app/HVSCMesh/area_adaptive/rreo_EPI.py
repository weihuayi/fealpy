import numpy as np
import gmsh
import matplotlib.pyplot as plt

#from TriRadiusRatio import TriRadiusRatio
from fealpy.mesh.mesh_quality import RadiusRatioQuality

from fealpy.mesh import TriangleMesh

mesh = TriangleMesh.from_meshio('area_EPI_adaptive.vtu')
node = mesh.entity('node')
cell = mesh.entity('cell')
node = node[:,:2]
mesh = TriangleMesh(node, cell)

angle = mesh.angle()
max_angle = np.max(angle,axis=1)
angles = max_angle*(180/np.pi)
m90 = np.sum(angles>90)
mesh.celldata['angles'] = angles

fig,axes1= plt.subplots()
axes1.set_ylim(0,8500)
fig.text(0.15,0.85,f'Number of angles>90:{m90}',fontsize=12)
mesh.show_angle(axes1,max_angle)
plt.show()

mesh.delete_degree_4()
isBdNode = mesh.ds.boundary_node_flag()
isFreeNode = ~isBdNode

node = mesh.entity('node')
cell = mesh.entity('cell')
# 去除度为4的点后进行优化
#opt = TriRadiusRatio(mesh)

from app.HVSCMesh.optimizer import *
from fealpy.mesh import TriangleMesh

mesh = TriangleMesh(node,cell)
mesh_quality = RadiusRatioQuality(mesh)
node = mesh.entity('node')
mesh = iterate_solver(mesh)

angle = mesh.angle()
max_angle = np.max(angle,axis=1)
angles = max_angle*(180/np.pi)
m90 = np.sum(angles>90)

fig,axes1= plt.subplots()
axes1.set_ylim(0,8500)
fig.text(0.15,0.85,f'Number of angles>90:{m90}',fontsize=12)
show_angle(axes1,max_angle)
plt.show()

mesh.celldata['angles'] = angles
mesh.to_vtk(fname='optEPI.vtu')


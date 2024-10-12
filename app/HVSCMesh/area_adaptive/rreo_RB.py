import numpy as np
import gmsh
import matplotlib.pyplot as plt

#from TriRadiusRatio import TriRadiusRatio
from fealpy.experimental.mesh.mesh_quality import RadiusRatioQuality

from fealpy.mesh import TriangleMesh

'''
gmsh.initialize()
gmsh.merge('adaptive_case_4_RB_IGCT.msh')

ntags, vxyz, _ = gmsh.model.mesh.getNodes()
node = vxyz.reshape((-1,3))
node = node[:,:2]
vmap = dict({j:i for i,j in enumerate(ntags)})
tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
evid = np.array([vmap[j] for j in evtags])
cell = evid.reshape((tris_tags.shape[-1],-1))
mesh = TriangleMesh(node,cell)
'''

mesh = TriangleMesh.from_meshio('area_RB_adaptive.vtu')
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
axes1.set_ylim(0,6000)
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

from optimizer import *
mesh = TriangleMesh(node,cell)
mesh_quality = RadiusRatioQuality(mesh)
node = mesh.entity('node')
mesh = iterate_solver(mesh)

angle = mesh.angle()
max_angle = np.max(angle,axis=1)
angles = max_angle*(180/np.pi)
m90 = np.sum(angles>90)

fig,axes1= plt.subplots()
axes1.set_ylim(0,6000)
fig.text(0.15,0.85,f'Number of angles>90:{m90}',fontsize=12)
show_angle(axes1,max_angle)
plt.show()

mesh.celldata['angles'] = angles
mesh.to_vtk(fname='optRB.vtu')


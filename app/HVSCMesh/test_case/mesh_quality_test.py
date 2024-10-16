import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.mesh.mesh_quality import RadiusRatioQuality

node = np.array([[0.0,0.0],[2.0,0.0],[1,np.sqrt(3)]],dtype=np.float64)
cell = np.array([[0,1,2]],dtype=np.int_)
mesh = TriangleMesh(node,cell)
mesh.uniform_refine(2)
node = mesh.entity('node')
cell = mesh.entity('cell')
node[cell[-1,0]] = node[cell[-1,0]]+[-0.15,0.05]
node[cell[-1,1]] = node[cell[-1,1]]+[-0.1,0.15]
node[cell[-1,2]] = node[cell[-1,2]]+[0,-0.15]

localEdge = mesh.localEdge
v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
area = np.cross(v[2],-v[1])/2

mesh_quality = RadiusRatioQuality(mesh) 

quality = mesh_quality(node)
print(quality)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes,aspect=1)
plt.show()


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh, TetrahedronMesh 


node = np.loadtxt('mesh/16/data/coor_d.dat')
cell = np.loadtxt('mesh/16/data/elem_d.dat', dtype=np.int)
cell = cell.reshape(-1, 4) - 1


mesh = TetrahedronMesh(node, cell)

a = mesh.entity_measure(3)
print(np.min(a))
face = mesh.entity(2)
idx = mesh.ds.boundary_face_index()
face = face[idx]

tmesh = TriangleMesh(node, face)

fig = plt.figure() 
axes = fig.add_subplot(111, projection='3d')
tmesh.add_plot(axes)
plt.show()

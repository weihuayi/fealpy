import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from scipy.spatial import Delaunay

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.mesh.TetrahedronMesh import TetrahedronMesh
from fealpy.mesh.TriangleMesh import TriangleMesh 


degree = 4
node = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1/2, np.sqrt(3)/2, 0],
    [1/2, np.sqrt(3)/6, np.sqrt(6)/3]], dtype=np.float)
cell = np.array([[0, 1, 2, 3]], dtype=np.int)

mesh = TetrahedronMesh(node, cell)

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.set_aspect('equal')
axes.set_axis_off()
edge0 = np.array([(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int) 
lines = a3.art3d.Line3DCollection(point[edge0], color='k', linewidths=2)
axes.add_collection3d(lines)
edge1 = np.array([(0, 2)], dtype=np.int)
lines = a3.art3d.Line3DCollection(point[edge1], color='gray', linewidths=2,
        alpha=0.5)
axes.add_collection3d(lines)
#mesh.add_plot(axes,  alpha=0.3)
mesh.find_node(axes, showindex=True, color='k', fontsize=20, markersize=100)


V = LagrangeFiniteElementSpace(mesh, degree)
ldof = V.number_of_local_dofs()
ipoints = V.interpolation_points()
cell2dof = V.dof.cell2dof[0]

ipoints = ipoints[cell2dof]

idx = np.arange(1, degree+2)
idx = np.cumsum(np.cumsum(idx))

d = Delaunay(ipoints)
mesh2 = TetrahedronMesh(ipoints, d.simplices)
face = mesh2.ds.face
isFace = np.zeros(len(face), dtype=np.bool_)
for i in range(len(idx)-1):
    flag = np.sum((face >= idx[i]) & (face < idx[i+1]), axis=-1) == 3
    isFace[flag] = True
face = face[isFace]

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.set_aspect('equal')
axes.set_axis_off()

edge0 = np.array([(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int) 
lines = a3.art3d.Line3DCollection(point[edge0], color='k', linewidths=2)
axes.add_collection3d(lines)
edge1 = np.array([(0, 2)], dtype=np.int)
lines = a3.art3d.Line3DCollection(point[edge1], color='gray', linewidths=2,
        alpha=0.5)
axes.add_collection3d(lines)

faces = a3.art3d.Poly3DCollection(ipoints[face], facecolor='w', edgecolor='k',
        linewidths=1, linestyle=':', alpha=0.3)
axes.add_collection3d(faces)


mesh.find_node(axes, node=ipoints, showindex=True, fontsize=20, color='r',
        markersize=100)
plt.show()

import numpy as np
from fealpy import function_space
from fealpy.Mesh import TetrahedronMesh

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

import sys


p = int(sys.argv[1])

#point = np.array([
#    [0, 0, 0],
#    [1, 0, 0],
#    [0, 1, 0],
#    [0, 0, 1],
#    [0, 0,-1]], dtype=np.float)
#
#cell = np.array([
#    [0, 1, 2, 3],
#    [0, 2, 1, 4]], dtype=np.int)

point = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]], dtype=np.float)

cell = np.array([[0, 1, 2, 3]], dtype=np.int)

mesh = TetrahedronMesh(point, cell)

V = function_space(mesh, 'Lagrange', p)
print(V.number_of_local_dofs())
print(V.number_of_global_dofs())
idx = V.cellIdx
for i in range(6):
    isEdgeDof = V.local_dof_on_edge(i)
    print(idx[isEdgeDof,:])

print("The interior dof in 4 faces:")
for i in range(4):
    isFaceDof = V.local_dof_on_face(i)
    print(idx[isFaceDof, :])

w = idx/p

point = np.tensordot(w, point[cell,:], axes=(1,1)).reshape(-1, 3, order='F')
print(point)

ax0 = a3.Axes3D(pl.figure())

mesh.add_plot(ax0)
ax0.plot(point[:, 0], point[:, 1], point[:,2], 'ro',markersize=20)
for i in range(point.shape[0]):
    ax0.text(point[i,0], point[i,1], point[i,2], str(i), fontsize=24) 

pl.show()


#print()
#for i in range(p, -1, -1):
#    m = p - i
#    for j in range(m, -1, -1):
#        n = m - j
#        for k in range(n, -1, -1):
#            q = n - k
#            print(i, end='\t')
#            print(j, end='\t')
#            print(k, end='\t')
#            print(q)
#

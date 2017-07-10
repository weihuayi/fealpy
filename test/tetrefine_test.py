

import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

from fealpy.mesh.TetrahedronMesh import TetrahedronMesh



#point = np.array([
#    [0, 0, 0],
#    [1, 0, 0],
#    [0, 1, 0],
#    [0, 0, 1]], dtype=np.float)
#
#cell = np.array([
#    [0, 1, 2, 3]], dtype=np.int)

point = np.array([
    [-1,-1,-1],
    [ 1,-1,-1], 
    [ 1, 1,-1],
    [-1, 1,-1],
    [-1,-1, 1],
    [ 1,-1, 1], 
    [ 1, 1, 1],
    [-1, 1, 1]], dtype=np.float) 

cell = np.array([
    [0,1,2,6],
    [0,5,1,6],
    [0,4,5,6],
    [0,7,4,6],
    [0,3,7,6],
    [0,2,3,6]], dtype=np.int)

mesh = TetrahedronMesh(point, cell)
for i in range(6):
    angle = mesh.dihedral_angle()
    print("max:", np.max(angle))
    print("min:", np.min(angle))
    mesh.uniform_refine()
ax0 = a3.Axes3D(pl.figure())
mesh.add_plot(ax0)
pl.show()

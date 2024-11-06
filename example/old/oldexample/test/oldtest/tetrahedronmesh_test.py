import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh import TetrahedronMesh
from fealpy.mesh.simple_mesh_generator import boxmesh3d

node = np.array([
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

mesh = TetrahedronMesh(node, cell)
mesh.uniform_refine(2)
mesh.print()
fig = pl.figure()
axes = a3.Axes3D(fig)
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
pl.show()



import numpy as np
import scipy.io as sio
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

from fealpy.mesh import TetrahedronMesh
from fealpy.mesh import TriangleMesh

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
mesh.uniform_refine(3)

face = mesh.entity('face')
isBdFace = mesh.ds.boundary_face_flag()

cell = face[isBdFace]
node = mesh.entity('node')
NN = mesh.number_of_nodes()
isBDNode = mesh.ds.boundary_node_flag()
NB = np.sum(isBDNode)
idxmap = np.zeros(NN, dtype=np.int32)
idxmap[isBDNode] = range(NB)
cell = idxmap[cell]
node = node[isBDNode]

trimesh = TriangleMesh(node, cell)

data = {
        'node':trimesh.node, 
        'elem':trimesh.ds.cell + 1 
        }

sio.matlab.savemat('trimesh', data)
ax0 = a3.Axes3D(pl.figure())
trimesh.add_plot(ax0)
pl.show()

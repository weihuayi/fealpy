import sys
import meshio
import numpy as np
#import scipy.io as sio

from fealpy.mesh import LagrangeTriangleMesh, MeshFactory, LagrangeWedgeMesh
from fealpy.writer import MeshWriter

#p = int(sys.argv[1])
#n = int(sys.argv[2])
#h = sys.argv[3]
#nh = int(sys.argv[4])

fname = 'initial/file1.vtu'
data = meshio.read(fname)
node = data.points
cell = data.cells[0][1]

mesh = LagrangeTriangleMesh(node*500, cell, p=1)
mesh.uniform_refine(n=0)
mesh = LagrangeWedgeMesh(mesh, h=0.005, nh=1, p=1)

edge = mesh.entity('edge')
node = mesh.entity('node')
print(np.mean(node, axis=0))
cell = mesh.entity('cell')
tface, qface = mesh.entity('face')

print(tface.shape)

mesh.celldata['a'] = np.arange(len(cell))
mesh.to_vtk(fname='write.vtu')




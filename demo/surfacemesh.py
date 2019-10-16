import sys
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.mesh.surface_mesh_generator import iso_surface
from fealpy.mesh.interface_mesh_generator import InterfaceMesh3d
from fealpy.mesh.level_set_function import Sphere, HeartSurface
from fealpy.mesh.SurfaceTriangleMeshOptAlg import SurfaceTriangleMeshOptAlg
from fealpy.mesh import TriangleMesh

#surface = Sphere()
surface = HeartSurface()
n = 20

#mesh = iso_surface(surface, surface.box, nx=n, ny=n, nz=n)
ialg = InterfaceMesh3d(surface, surface.box, n)
mesh = ialg.run('interfacemesh')

alg = SurfaceTriangleMeshOptAlg(surface, mesh, gamma=1, theta=0.1)

isNonDelaunayEdge = alg.run(100)

NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()
cell = mesh.entity('cell')
print(NN, cell.max(), cell.min())
node = mesh.entity('node')
edge2cell = mesh.ds.edge_to_cell()
isNonDelaunayCell = np.zeros(NC, dtype=np.bool)
isNonDelaunayCell[edge2cell[isNonDelaunayEdge, 0:2]] = True

mesh0 = TriangleMesh(node, cell[isNonDelaunayCell])

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
plt.show()

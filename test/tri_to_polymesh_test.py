
import sys
import matplotlib.pyplot as plt

from fealpy.mesh.simple_mesh_generator import unitcircledomainmesh
from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityNode
from fealpy.mesh.PolygonMesh import PolygonMesh


h = 0.1
mesh0 = unitcircledomainmesh(h) 

mesh1 = TriangleMeshWithInfinityNode(mesh0)

point, cell, cellLocation = mesh1.to_polygonmesh()

pmesh = PolygonMesh(point, cell, cellLocation)

f0 = plt.figure()
axes0 = f0.gca()
mesh0.add_plot(axes0)

f1 = plt.figure()
axes1 = f1.gca()
pmesh.add_plot(axes1)
plt.show()

import numpy as np

import matplotlib.pyplot as plt
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityPoint 
from fealpy.mesh.PolygonMesh import PolygonMesh 


# Generate mesh
box = [-1, 1, -1, 1]
mesh = rectangledomainmesh(box, nx=10, ny=10, meshtype='tri')
mesht = TriangleMeshWithInfinityPoint(mesh)
ppoint, pcell, pcellLocation = mesht.to_polygonmesh()
pmesh = PolygonMesh(ppoint, pcell, pcellLocation) 

# Virtual  element space 

fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
plt.show()

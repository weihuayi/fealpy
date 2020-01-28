#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.simple_mesh_generator import distmesh2d
from fealpy.geometry import dcircle, drectangle, ddiff
from fealpy.geometry import DistDomain2d
from fealpy.geometry import huniform

from fealpy.mesh import DistMesh2d
from fealpy.mesh import PolygonMesh
from fealpy.mesh import TriangleMeshWithInfinityNode

fd = lambda p: drectangle(p, [0.0, 1.0, 0.0, 1.0])
fh = huniform
bbox = [-0.2, 1.2, -0.2, 1.2]
pfix = np.array([
    (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
h0 = 0.05
domain = DistDomain2d(fd, fh, bbox, pfix)
distmesh2d = DistMesh2d(domain, h0)
distmesh2d.run()
mesh = TriangleMeshWithInfinityNode(distmesh2d.mesh)
pnode, pcell, pcellLocation = mesh.to_polygonmesh()
pmesh = PolygonMesh(pnode, pcell, pcellLocation)

fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)

fig = plt.figure()
axes = fig.gca()
distmesh2d.mesh.add_plot(axes)
plt.show()


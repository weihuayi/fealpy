#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import PolygonMesh
from fealpy.functionspace import NCVEMDof2d, NonConformingVirtualElementSpace2d
from fealpy.mesh.simple_mesh_generator import unitcircledomainmesh


# mesh = unitcircledomainmesh(0.3, meshtype='polygon')


node = np.array([
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
    (1.0, 0.5),
    (0.0, 1.0),
    (0.5, 1.0),
    (1.0, 1.0)], dtype=np.float)

cell = np.array(
        [1, 4, 0, 3, 0, 4, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7], dtype=np.int)
cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

mesh = PolygonMesh(node, cell, cellLocation)


dof = NCVEMDof2d(mesh, p=4)
ipoints = dof.interpolation_points()

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ipoints, showindex=True)
mesh.find_node(axes, showindex=True)
plt.show()

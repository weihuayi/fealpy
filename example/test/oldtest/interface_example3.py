import sys

import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from fealpy.mesh.level_set_function import dtheta
from fealpy.mesh.interface_mesh_generator import interfacemesh2d
from fealpy.mesh.TriangleMesh import TriangleMesh 

n = int(sys.argv[1])

box = [-25, 25, -25, 25]

phi = lambda p:dtheta(p)
pmesh = interfacemesh2d(box, phi, n)

a = pmesh.angle()
print(np.max(a))
fig = plt.figure()
axes = fig.gca() 
pmesh.add_plot(axes, cellcolor='w')
#pmesh.find_point(axes, color=pmesh.pointMarker, markersize=20)
fig.savefig('mesh.png')
plt.show()

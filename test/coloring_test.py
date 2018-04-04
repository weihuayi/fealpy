import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh.simple_mesh_generator import unitcircledomainmesh
from fealpy.mesh.coloring import coloring

fig = plt.figure()
axes = fig.gca()
mesh = unitcircledomainmesh(0.1)
c = coloring(mesh, method='random')
color=np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
mesh.add_plot(axes, nodecolor=color[c-1], cellcolor='w', markersize=100)
mesh.find_node(axes, color=color[c-1])
plt.show()

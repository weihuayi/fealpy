
import numpy as np
import matplotlib.pyplot as plt
import sys

from fealpy.mesh.simple_mesh_generator import rectangledomainmesh

box = [0, 1, 0, 1]
qmesh = rectangledomainmesh(box, nx=2, ny=2, meshtype='quad')

qmesh.print()

fig = plt.figure()
axes = fig.gca()
qmesh.add_plot(axes)
qmesh.find_point(axes, showindex=True)
qmesh.find_edge(axes)
qmesh.find_cell(axes, showindex=True)
plt.show()

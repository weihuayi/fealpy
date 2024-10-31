import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh.level_set_function import dsine
from fealpy.mesh.interface_mesh_generator import interfacemesh2d

n = int(sys.argv[1])

box = [-5, 5, -5, 5]

cxy = (0.0, 0.0)
r = 2

phi = lambda p: dsine(p, cxy, r)
mesh = interfacemesh2d(box, phi, n)


fig = plt.figure()
axes = fig.gca() 
mesh.add_plot(axes, pointcolor=mesh.pointMarker)
plt.show()

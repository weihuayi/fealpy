import sys

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh.level_set_function import dtheta
from fealpy.mesh.interface_mesh_generator import interfacemesh2d

import time


box = [-1, 1, -1, 1]
cxy = (0.0, 0.0)
r = 0.5
phi = lambda p: dcircle(p, cxy, r)

n = [100, 200, 400, 800, 1600]
for i in n:
    b0 = time.clock()
    mesh = interfacemesh2d(box, phi, i)
    b1 = time.clock()
    print("time:", b1 - b0)
    a = mesh.angle()
    print(np.max(a))

box = [-25, 25, -25, 25]
phi = dtheta
for i in n:
    b0 = time.clock()
    mesh = interfacemesh2d(box, phi, i)
    b1 = time.clock()
    print("time:", b1 - b0)
    a = mesh.angle()
    print(np.max(a))


#circle = Circle(cxy, r, edgecolor='g', fill=False, linewidth=2)
#fig = plt.figure()
#axes = fig.gca()
#mesh.add_plot(axes, pointcolor=mesh.pointMarker) 
#axes.add_patch(circle)
#plt.show()


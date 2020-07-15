#!/usr/bin/env python3
# 
import sys 
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh import MeshFactory

mf = MeshFactory()

box = [0, 1, 0, 1]
mesh = mf.boxmesh2d(box, nx=10, ny=10, meshtype='tri')



# plot 
GD = mesh.geo_dimension()
fig = plt.figure()
if GD == 2:
    axes = fig.gca() 
    mesh.add_plot(axes)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True)
elif GD == 3:
    axes = fig.gca(projection='3d')
    mesh.add_plot(axes)

plt.show()

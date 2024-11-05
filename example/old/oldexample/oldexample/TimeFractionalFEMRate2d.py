#!/usr/bin/env python3
#

import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.time_fractional_2d import FisherData2d
from mpl_toolkits.mplot3d import Axes3D

pde = FisherData2d()

mesh = pde.init_mesh(n=5)
timeline = pde.time_mesh(0, 1, 100)

node = mesh.entity('node')
cell = mesh.entity('cell')

uI = pde.solution(node, 1.0)

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot_trisurf(node[:, 0], node[:, 1], cell, uI, cmap=plt.cm.jet, lw=0.0)

plt.show()


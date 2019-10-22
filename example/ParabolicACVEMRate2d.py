import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.parabolic_model_2d import SpaceMeasureDiracSourceData


pde = SpaceMeasureDiracSourceData()
mesh = pde.init_mesh(n=8)
timeline = pde.time_mesh(0, 1, 100)

node = mesh.entity('node')
cell = mesh.entity('cell')

uI = pde.solution(node, 1.0)

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot_trisurf(node[:, 0], node[:, 1], cell, uI, cmap=plt.cm.jet, lw=0.0)

plt.show()

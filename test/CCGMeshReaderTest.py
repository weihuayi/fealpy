


import sys
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh import CCGMeshReader



fname = sys.argv[1]
reader = CCGMeshReader(fname)
mesh = reader.read()

node = mesh.entity('node')
cell = mesh.entity('cell')


fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot_trisurf(node[:, 0], node[:, 1], node[:, 2], triangles=cell)
plt.show()



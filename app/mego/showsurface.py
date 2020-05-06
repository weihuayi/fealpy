import sys

import numpy as np
import scipy.io as sio

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh

f = sys.argv[1]

data = sio.loadmat(f)

node = data['node']
cell = data['elem']-1

mesh = TriangleMesh(node, cell)


fig = plt.figure()
axes = Axes3D(fig) 
mesh.add_plot(axes)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import pylab as pl
import scipy.io as sio

from fealpy.mesh import simple_mesh_generator, TriangleMesh


mesh = simple_mesh_generator.unitcircledomainmesh(0.5)
node = mesh.entity('node')
cell = mesh.entity('cell')


print(node.shape)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()


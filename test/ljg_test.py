import sys


import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh.level_set_function import Sphere, TorusSurface, EllipsoidSurface, HeartSurface, OrthocircleSurface, QuarticsSurface
from fealpy.mesh.TriangleMesh import TriangleMesh 

n = int(sys.argv[1])

surface = Sphere()
mesh = surface.init_mesh()
mesh.uniform_refine(n=n, surface=surface)
node = mesh.node
cell = mesh.ds.cell
mesh = TriangleMesh(node, cell)


#fig = plt.figure()
#axes = fig.gca(projection='3d')
#mesh.add_plot(axes)
#x = mesh.node[:, 0]
#y = mesh.node[:, 1]
#z = mesh.node[:, 2]
#axes.plot_trisurf(x, y, z, triangles=mesh.ds.cell, cmap=plt.cm.jet,lw=2.0)
#plt.show()

fig = pl.figure() 
axes = a3.Axes3D(fig)
mesh.add_plot(axes)
pl.show()

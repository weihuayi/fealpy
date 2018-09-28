import sys
import numpy as np
import mpl_toolkits.mplot3d as a3
import pylab as pl
from fealpy.mesh.level_set_function import Sphere, HeartSurface
from fealpy.mesh.TriangleMesh import TriangleMesh

m = int(sys.argv[1])
surface = Sphere()
mesh = surface.init_mesh()
mesh.uniform_refine(n=m,surface = surface)
node = mesh.node
cell = mesh.ds.cell
mesh = TriangleMesh(node, cell)
fig = pl.figure()
axes = a3.Axes3D(fig)
mesh.add_plot(axes)
pl.show()

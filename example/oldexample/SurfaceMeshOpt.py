import sys
import numpy as np
import mpl_toolkits.mplot3d as a3

import pylab as pl

from mayavi import mlab

from fealpy.mesh.surface_mesh_generator import iso_surface
from fealpy.mesh.level_set_function import Sphere
from fealpy.mesh.TriangleMesh import TriangleMesh 

surface = Sphere()
mesh = surface.init_mesh()
mesh.uniform_refine(3, surface)

c, R = mesh.circumcenter()

f = pl.figure()
axes = a3.Axes3D(f)
mesh.add_plot(axes, showaxis=True)
mesh.find_node(axes, node=c, markersize=300)
pl.show()



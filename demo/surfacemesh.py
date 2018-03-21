import sys
import numpy as np

import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh.surface_mesh_generator import iso_surface
from fealpy.mesh.level_set_function import Sphere, HeartSurface

#surface = Sphere()
surface = HeartSurface()
n = 20
mesh = iso_surface(surface, surface.box, nx=n, ny=n, nz=n)

fig = pl.figure() 
axes = a3.Axes3D(fig)
mesh.add_plot(axes)
pl.show()

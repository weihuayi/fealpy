import sys

import numpy as np

import mpl_toolkits.mplot3d as a3
import pylab as pl

from mayavi import mlab

from fealpy.mesh.surface_mesh_generator import iso_surface
from fealpy.mesh.level_set_function import Sphere, TwelveSpheres, HeartSurface
from fealpy.mesh.level_set_function import OrthocircleSurface, QuarticsSurface
from fealpy.mesh.level_set_function import TorusSurface, EllipsoidSurface

surface = TwelveSpheres()
surface = HeartSurface()
surface = OrthocircleSurface()
surface = QuarticsSurface()
surface = TorusSurface()
surface = EllipsoidSurface()
n = 40 
mesh = iso_surface(surface, surface.box, nx=n, ny=n, nz=n)


f = pl.figure()
axes = a3.Axes3D(f)
mesh.add_plot(axes, showaxis=True)
pl.show()

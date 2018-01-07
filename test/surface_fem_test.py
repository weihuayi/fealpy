import sys

import numpy as np

import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh.level_set_function import Sphere
from fealpy.functionspace.surface_lagrange_fem_space import SurfaceTriangleMesh
p = int(sys.argv[1])
r = int(sys.argv[2])
surface = Sphere()
smesh = surface.init_mesh()
smesh.uniform_refine(r, surface)

stmesh = SurfaceTriangleMesh(smesh, surface, p)
a = stmesh.einsum_area()
print(smesh.area().sum())
print('{:2.16f}'.format(a.sum()))
print(4*np.pi)
f = pl.figure()
axes = a3.Axes3D(f)
smesh.add_plot(axes)
smesh.find_points(axes, point=stmesh.point)
pl.show()




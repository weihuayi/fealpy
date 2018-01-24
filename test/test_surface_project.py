import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.level_set_function import Sphere, TorusSurface, EllipsoidSurface,HeartSurface, OrthocircleSurface, QuarticsSurface
from fealpy.functionspace.surface_lagrange_fem_space import SurfaceLagrangeFiniteElementSpace
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace 

p=3
#surface = OrthocircleSurface()
surface = HeartSurface()
mesh = surface.init_mesh()
mesh.uniform_refine(2, surface=surface)
#V = LagrangeFiniteElementSpace(mesh, p)
#ipoint = V.interpolation_points()
#point, d = surface.project(ipoint)

fig = plt.figure()
fig.set_facecolor('white')
axes = fig.gca(projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
z = mesh.point[:, 2]
axes.plot_trisurf(x, y, z, triangles=mesh.ds.cell, cmap=plt.cm.jet,lw=2.0)
plt.show()

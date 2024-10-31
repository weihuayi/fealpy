import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import pylab as pl
from fealpy.geometry import Circle
from fealpy.geometry import Sphere, HeartSurface, EllipsoidSurface,TorusSurface, OrthocircleSurface, QuarticsSurface




surface =  HeartSurface()
mesh = surface.init_mesh()
mesh.uniform_refine(n=0, surface=surface)
node = mesh.entity('node')
cell = mesh.entity('cell')

fig = pl.figure()
axes = a3.Axes3D(fig)
mesh.add_plot(axes)
pl.show()






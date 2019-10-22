import sys
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import pylab as pl
from fealpy.mesh.level_set_function import Sphere
from fealpy.mesh.Tritree import Tritree


surface = Sphere()
mesh = surface.init_mesh()
mesh.uniform_refine(n=7, surface = surface)
node = mesh.entity('node')
cell = mesh.entity('cell')
h = mesh.entity_measure('edge')
print(node.shape)
print('h:', h)
fig = pl.figure()
axes = a3.Axes3D(fig)
mesh.add_plot(axes, cellcolor='w', linewidths=1)

pl.show()

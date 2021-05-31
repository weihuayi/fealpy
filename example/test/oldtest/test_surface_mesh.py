import numpy as np
import scipy.io as sio
from fealpy.mesh import TriangleMesh
import mpl_toolkits.mplot3d as a3
import pylab as pl
from fealpy.mesh.implicit_surface import HeartSurface

heart = HeartSurface()
mesh = heart.init_mesh('/home/why/heart1.mat')
node = mesh.entity('node')
cell = mesh.entity('cell')

node, _ = heart.project(node, maxit=1)

mesh.node = node


mesh.uniform_refine(n=2, surface=heart)
mesh.uniform_refine(n=1)

fig = pl.figure()
axes = a3.Axes3D(fig)
mesh.add_plot(axes)

pl.show()

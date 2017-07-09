


import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

from fealpy.mesh.TetrahedronMesh import TetrahedronMesh



point = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0,-1]], dtype=np.float)

cell = np.array([
    [0, 1, 2, 3],
    [2, 1, 0, 4]], dtype=np.int)

mesh = TetrahedronMesh(point, cell)
mesh.uniform_refine()
ax0 = a3.Axes3D(pl.figure())
mesh.add_plot(ax0)
pl.show()

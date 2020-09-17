

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import collections  as mc

from fealpy.mesh import MeshFactory

mf = MeshFactory()
box = [0, 1, 0, 1]

mesh = mf.boxmesh2d(box, nx=5, ny=5, meshtype='tri')

point = np.array([
    (0.10, 0.50),
    (0.90, 0.50),
    (0.15, 0.10),
    (0.31, 0.90)], dtype=np.float64)
segment = np.array([(0, 1), (2, 3)], dtype=np.int_)

isCutCell = mesh.find_crossed_cell(point, segment)

mesh.bisect(isCutCell)
#isCutCell = mesh.find_crossed_cell(point, segment)
#mesh.bisect(isCutCell)
#isCutCell = mesh.find_crossed_cell(point, segment)


lc = mc.LineCollection(point[segment], linewidths=2)
cidx, = np.nonzero(isCutCell)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, showindex=True)
axes.add_collection(lc)
plt.show()


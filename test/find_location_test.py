

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory

mf = MeshFactory()
box = [0, 1, 0, 1]

mesh = mf.boxmesh2d(box, nx=5, ny=5, meshtype='tri')

point = np.array([
    (0.1, 0.1),
    (0.8, 0.8),
    (0.1, 0.8),
    (0.8, 0.1)], dtype=np.float64)
segment = np.array([(0, 1), (2, 3)], dtype=np.int_)


for i in range(8):
    isCutCell= mesh.find_segment_location(point, segment)
    mesh.bisect(isCutCell)

isCutCell = mesh.find_segment_location(point, segment)
cidx, = np.nonzero(isCutCell)

node = mesh.entity('node')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, index=cidx)
plt.show()




import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory

mf = MeshFactory()
box = [0, 1, 0, 1]

mesh = mf.boxmesh2d(box, nx=5, ny=5, meshtype='tri')

node = np.array([
    (0.1, 0.1),
    (0.8, 0.8)], dtype=np.float64)
edge = np.array([(0, 1)], dtype=np.int_)

cidx = mesh.find_segment_location(node, edge)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, index=cidx, showindex=True)
mesh.find_node(axes, node=node, showindex=True)
plt.show()


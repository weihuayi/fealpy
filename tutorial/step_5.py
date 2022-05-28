import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF

box = 2*[0, 1]

mesh = MF.boxmesh2d(box, nx=2, ny=2, meshtype='tri')

qf = mesh.integrator(3, etype='edge')
# bcs.shape == (NQ, 2)
# ws.shape == (NQ, )
bcs, ws = qf.get_quadrature_points_and_weights()
quadpts = mesh.bc_to_point(bcs) # (NQ, NE, 2)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=quadpts.reshape(-1, 2), markersize=25)
plt.show()

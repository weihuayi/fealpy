import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF

box = 2*[0, 1]

mesh = MF.boxmesh2d(box, nx=2, ny=2, meshtype='quad')

qf = mesh.integrator(2, etype='edge')
bcs, ws = qf.get_quadrature_points_and_weights()
ps = mesh.bc_to_point(bcs) 

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ps.reshape(-1, 2), markersize=25)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.spatial import Delaunay
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.mesh.TriangleMesh import TriangleMesh

degree = 3

fig = plt.figure()

point = np.array([
    [0,0],
    [1,0],
    [1/2,np.sqrt(3)/2]], dtype=np.float)
cell = np.array([[0, 1, 2]], dtype=np.int)

mesh = TriangleMesh(point, cell)

axes = fig.add_subplot(1, 2, 1)
axes.set_aspect('equal')
axes.set_axis_off()
mesh.add_plot(axes, cellcolor='w')
mesh.find_point(axes, showindex=True, color='k', fontsize=12, markersize=25)

V = LagrangeFiniteElementSpace(mesh, degree)
ldof = V.number_of_local_dofs()
ipoints = V.interpolation_points()
cell2dof = V.dof.cell2dof[0]
ipoints = ipoints[cell2dof]


d = Delaunay(ipoints)
mesh2 = TriangleMesh(ipoints, d.simplices)

axes = fig.add_subplot(1, 2, 2)
axes.set_aspect('equal')
axes.set_axis_off()
edge = mesh2.ds.edge

lines = LineCollection(ipoints[edge], color='k', linewidths=1, linestyle=':')
axes.add_collection(lines)

mesh.add_plot(axes, cellcolor='w', linewidths=1)
mesh.find_point(axes, point=ipoints, showindex=True, fontsize=12, markersize=25)
plt.savefig('/home/why/tridof4.pdf')
plt.show()

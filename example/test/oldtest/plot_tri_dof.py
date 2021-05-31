import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.spatial import Delaunay
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.mesh.TriangleMesh import TriangleMesh

degree = 4


point = np.array([
    [0,0],
    [1,0],
    [1/2,np.sqrt(3)/2]], dtype=np.float)
cell = np.array([[0, 1, 2]], dtype=np.int)

mesh = TriangleMesh(point, cell)

fig = plt.figure()
axes = fig.gca()
axes.set_aspect('equal')
axes.set_axis_off()
mesh.add_plot(axes, cellcolor='w', linewidths=2)
mesh.find_point(axes, showindex=True, color='k', fontsize=20, markersize=25)

V = LagrangeFiniteElementSpace(mesh, degree)
ldof = V.number_of_local_dofs()
ipoints = V.interpolation_points()
cell2dof = V.dof.cell2dof[0]
ipoints = ipoints[cell2dof]


d = Delaunay(ipoints)
mesh2 = TriangleMesh(ipoints, d.simplices)

fig = plt.figure()
axes = fig.gca()
axes.set_aspect('equal')
axes.set_axis_off()
edge = mesh2.ds.edge

lines = LineCollection(ipoints[edge], color='k', linewidths=1, linestyle=':')
axes.add_collection(lines)

mesh.add_plot(axes, cellcolor='w', linewidths=2)
mesh.find_point(axes, point=ipoints, showindex=True, fontsize=20, markersize=25)
plt.show()

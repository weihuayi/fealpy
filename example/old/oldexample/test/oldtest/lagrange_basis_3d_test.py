import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

import sys

from fealpy.mesh.TetrahedronMesh import TetrahedronMesh
from fealpy.functionspace.tools import function_space

degree = int(sys.argv[1])

ax0 = a3.Axes3D(pl.figure())

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
V = function_space(mesh, 'Lagrange', degree)
cell2dof = V.cell_to_dof()
ipoints = V.interpolation_points()
mesh.add_plot(ax0)
mesh.find_point(ax0, point=ipoints, showindex=True)
mesh.print()
print('cell2dof:\n', cell2dof)
pl.show()


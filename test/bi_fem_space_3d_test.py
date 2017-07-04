
import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

import sys

from fealpy.mesh.tree_data_structure import Octree
from fealpy.mesh.HexahedronMesh import HexahedronMesh 

from fealpy.functionspace.tools import function_space

degree = int(sys.argv[1])


point = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]], dtype=np.float)

cell = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int)

tree = Octree(point, cell)
tree.uniform_refine()
isLeafCell = tree.is_leaf_cell()
print('child', tree.child)
print('isLeafCell', isLeafCell)

mesh = HexahedronMesh(tree.point, tree.ds.cell[isLeafCell])

V = function_space(mesh, 'Q', degree)
cell2dof = V.cell_to_dof()
ipoints = V.interpolation_points()

ax0 = a3.Axes3D(pl.figure())
ax1 = a3.Axes3D(pl.figure())
mesh.add_plot(ax0)
mesh.find_point(ax0,  showindex=True)

mesh.add_plot(ax1)
mesh.find_point(ax1, point=ipoints,  showindex=True)
mesh.print()
print('cell2dof:\n', cell2dof)
pl.show()

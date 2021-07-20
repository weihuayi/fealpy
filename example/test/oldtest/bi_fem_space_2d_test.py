

import numpy as np
import matplotlib.pyplot as plt
import sys

from fealpy.mesh.tree_data_structure import Quadtree
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh  

from fealpy.functionspace.tools import function_space

degree = int(sys.argv[1])


point = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]], dtype=np.float)

cell = np.array([[0, 1, 2, 3]], dtype=np.int)

tree = Quadtree(point, cell)
tree.uniform_refine()
isLeafCell = tree.is_leaf_cell()

mesh = QuadrangleMesh(tree.point, tree.ds.cell[isLeafCell])

V = function_space(mesh, 'Q', degree)
cell2dof = V.cell_to_dof()
ipoints = V.interpolation_points()

fig0 = plt.figure()
ax0 = fig0.gca()
mesh.add_plot(ax0)
mesh.find_point(ax0,  showindex=True)

fig1 = plt.figure()
ax1 = fig1.gca()
mesh.add_plot(ax1)
mesh.find_point(ax1, point=ipoints,  showindex=True)
mesh.print()
print('cell2dof:\n', cell2dof)
plt.show()

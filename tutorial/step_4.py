
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace

box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh, p=4, spacetype='C')

ipoints = space.interpolation_points()
cell2dof = space.cell_to_dof() # (NC, 10)
print(cell2dof)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=25)
mesh.find_edge(axes, showindex=True, fontsize=30)
mesh.find_cell(axes, showindex=True, fontsize=50)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ipoints, showindex=True, fontsize=25)
plt.show()

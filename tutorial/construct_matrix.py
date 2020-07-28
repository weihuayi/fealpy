import sys
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace


p = int(sys.argv[1])

mf = MeshFactory()
box = [0, 1, 0, 1]
mesh = mf.boxmesh2d(box, nx=1, ny=1, meshtype='tri')
space = LagrangeFiniteElementSpace(mesh, p=p)

ipoints = space.interpolation_points() # (gdof, 2)
cell2dof = space.cell_to_dof() 
print('cell2dof:')
for i, val in enumerate(cell2dof):
    print(i, ": ", val)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=24)
mesh.find_edge(axes, showindex=True, fontsize=22)
mesh.find_cell(axes, showindex=True, fontsize=20)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ipoints, showindex=True, color='r', fontsize=24)
plt.show()

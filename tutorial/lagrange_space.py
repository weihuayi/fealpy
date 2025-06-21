
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace

box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

cell = mesh.entity('cell')
edge = mesh.entity('edge')
print('cell:\n', cell)

space1 = LagrangeFiniteElementSpace(mesh, p=1)
cell2dof = space1.cell_to_dof() # (NC, ldof) 
ips = space1.interpolation_points()
print('cell2dof:\n', cell2dof)
print('ips:\n', ips)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=50)
mesh.find_edge(axes, showindex=True, fontsize=55)
mesh.find_cell(axes, showindex=True, fontsize=60)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ips, showindex=True, 
        fontsize=30, markersize=50)
plt.show()

if False:
    space2 = LagrangeFiniteElementSpace(mesh, p=2)
    cell2dof = space2.cell_to_dof()
    ips = space2.interpolation_points()
    print('cell2dof:\n', cell2dof)
    print('ips:\n', ips)

    space3 = LagrangeFiniteElementSpace(mesh, p=3)
    cell2dof = space3.cell_to_dof()
    ips = space3.interpolation_points()
    print('cell2dof:\n', cell2dof)
    print('ips:\n', ips)
    print('edge:\n', edge)

    space5 = LagrangeFiniteElementSpace(mesh, p=5)
    cell2dof = space5.cell_to_dof()
    ips = space5.interpolation_points()
    print('cell2dof:\n', cell2dof)
    print('ips:\n', ips)
    print('edge:\n', edge)


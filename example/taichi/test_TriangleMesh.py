import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

from fealpy.mesh import MeshFactory as MF
from fealpy.ti import TriangleMesh # 基于 Taichi 的三角形网格

from fealpy.functionspace import LagrangeFiniteElementSpace as LFESpace

ti.init()


p = 2
node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)
mesh = TriangleMesh(node, cell)

mi = mesh.multi_index_matrix(p)
print(mi)


NP = mesh.number_of_global_interpolation_points(p)
ipoints = ti.field(ti.f64, shape=(NP, 2))
mesh.interpolation_points(p, ipoints)
print(NP)
print(ipoints)


NC = mesh.number_of_cells()
ldof = (p+1)*(p+2)//2
cell2dof = ti.field(ti.u32, shape=(NC, ldof))
mesh.cell_to_dof(p, cell2dof)
print(cell2dof)
print(mesh.cell2edge)

mesh = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri')
space = LFESpace(mesh, p)
cell2dof = space.cell_to_dof()
ips = space.interpolation_points()
print(cell2dof)
print(mesh.ds.cell_to_edge())
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, showindex=True)
mesh.find_node(axes, node=ipoints.to_numpy(), showindex=True)
plt.show()












import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

from fealpy.mesh import MeshFactory as MF
from fealpy.ti import TriangleMesh # 基于 Taichi 的三角形网格

ti.init()


p = 3
node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)
mesh = TriangleMesh(node, cell)


NC = mesh.number_of_cells()
NP = mesh.number_of_interpolation_points(p)

ipoints = ti.field(ti.f64, shape=(NP, 2))
ldof = (p+1)*(p+2)//2
cell2dof = ti.field(ti.u32, shape=(NC, ldof))
mesh.cell_to_dof(p, cell2dof)

mesh = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri')
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node)
plt.show()












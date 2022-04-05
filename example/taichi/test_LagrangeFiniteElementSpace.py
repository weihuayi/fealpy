
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF

import taichi as ti
from fealpy.ti import TriangleMesh 
from fealpy.ti import TetrahedronMesh
from fealpy.ti import LagrangeFiniteElementSpace
from fealpy.functionspace import LagrangeFiniteElementSpace as LFESpace

ti.init()

node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)

mesh = TriangleMesh(node, cell)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

print("tiedge:", mesh.edge)
print("ticell2edge:", mesh.cell2edge)

space = LagrangeFiniteElementSpace(mesh, p=3)

bc = np.array([1/3, 1/3, 1/3], dtype=np.float64)
space.pytest(bc)

print(space.multiIndex)
print(space.geo_dimension())
print(space.top_dimension())
print(space.number_of_local_dofs())

mesh = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri')

print("egde:", mesh.ds.edge)
print("cell2edge:", mesh.ds.cell_to_edge())

lspace = LFESpace(mesh, p=3)

cell2dof = lspace.cell_to_dof()
ips = lspace.interpolation_points()
print("cell2dof:", cell2dof)
print("edge2dof:", lspace.edge_to_dof())

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ips, showindex=True)
plt.show()




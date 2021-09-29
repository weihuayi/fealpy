import numpy as np
import matplotlib.pyplot as plt
import copy

from fealpy.mesh import HalfEdgeMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace

# 生成网格
node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float_)
cell = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int_)
mesh0 = TriangleMesh(node, cell) 
mesh0 = HalfEdgeMesh2d.from_mesh(mesh0)
mesh0.init_level_info()

isMarked = np.array([0, 1, 0], dtype=np.bool_)
mesh0.refine_triangle_rg(isMarked)

# 生成空间
space0 = LagrangeFiniteElementSpace(mesh0, p=1)

# 生成加密后网格的空间
mesh1 = copy.deepcopy(mesh0)
isMarked = np.array([0, 1, 0, 0, 0, 1, 1], dtype=np.bool_)
mesh1.refine_triangle_rg(isMarked)
space1 = LagrangeFiniteElementSpace(mesh1, p=1)

# 生成粗化后网格的空间
mesh2 = copy.deepcopy(mesh1)
isMarked = np.zeros(mesh2.number_of_all_cells(), dtype=np.bool_)
isMarked[[5, 8, 10, 12, 13, 16]] = True
mesh2.coarsen_triangle_rg(isMarked)
space2 = LagrangeFiniteElementSpace(mesh2, p=1)

uh0 = space0.function()
uh1 = space1.function()
uh2 = space2.function()

#粗网格到细网格
NN0 = mesh0.number_of_nodes()

uh0[:] = np.array([1, 2, 3 ,4 ,5 ,6 ,7])
uh1[:NN0] = uh0

nn2e = mesh1.newnode2edge
edge = mesh0.entity("edge")

uh1[NN0:] = np.average(uh0[edge[nn2e]], axis=-1)
print(uh1)

# 细网格到粗网格
retain = mesh2.retainnode
uh2[:] = uh1[retain]

fig = plt.figure()
axes = fig.gca(projection='3d')
uh0.add_plot(axes)

fig = plt.figure()
axes = fig.gca(projection='3d')
uh1.add_plot(axes)

fig = plt.figure()
axes = fig.gca(projection='3d')
uh2.add_plot(axes)

fig = plt.figure()
axes = fig.gca()
mesh0.add_plot(axes)
mesh0.find_node(axes, showindex=True)
mesh0.find_edge(axes, showindex=True)

fig = plt.figure()
axes = fig.gca()
mesh1.add_plot(axes)
mesh1.find_node(axes, showindex=True)
mesh1.find_cell(axes, showindex=True)

fig = plt.figure()
axes = fig.gca()
mesh2.add_plot(axes)
mesh2.find_node(axes, showindex=True)
mesh2.find_cell(axes, showindex=True)
plt.show()



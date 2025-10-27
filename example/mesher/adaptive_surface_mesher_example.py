import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from fealpy.backend import bm
from fealpy.mesh import TriangleMesh
from fealpy.mesher import SphereSurfaceMesher
from fealpy.mmesh.tool import high_order_meshploter

from aabbtree import AABBTree

bm.set_backend('numpy')


# 生成二次曲面单元网格
mesher = SphereSurfaceMesher()
mesh = mesher.init_mesh(2)
mesh.uniform_refine(3)

ps = mesh.entity_barycenter('cell')


# 生成对应的线性网格单元
isCornerNode = mesh.cell_corner_node_flag()
node = mesh.entity('node')[isCornerNode]
cell = mesh.entity('cell')[:, [0, 3, 5]]
NN  = sum(isCornerNode)
idmap = bm.zeros(mesh.number_of_nodes(), dtype=bm.int32)
idmap = bm.set_at(idmap, bm.where(isCornerNode)[0], bm.arange(NN, dtype=bm.int32))
cell = idmap[cell]

linearMesh = TriangleMesh(node, cell)


print(cell.dtype)

tree = AABBTree(node, cell)

idxs, bc = tree.query(ps)
print(idxs.dtype)
print(bc.dtype)

ps = mesh.bc_to_point(bc, idxs, map_mode='pair')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
linearMesh.add_plot(ax)
linearMesh.find_node(ax, node=ps)
#high_order_meshploter(ax, mesh)
plt.show()


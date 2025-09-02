import matplotlib.pyplot as plt

from fealpy.backend import bm
from fealpy.geometry.implicit_surface import SphereSurface
from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh
from fealpy.mesh import QuadrangleMesh, LagrangeQuadrangleMesh
from fealpy.mmesh.tool import high_order_meshploter

bm.set_backend('numpy')

def plot_nodes(ax, node, fontsize=10, color='k'):
    """
    """
    ax.scatter(node[:, 0], node[:, 1], color="blue", s=30, zorder=2)

    # 标记节点的行号
    for i, (x, y) in enumerate(node):
        ax.text(x, y, str(i), fontsize=fontsize, color=color)

# surface = SphereSurface()
# qmesh = QuadrangleMesh.from_unit_sphere_surface()
# mesh = LagrangeQuadrangleMesh.from_quadrangle_mesh(qmesh, p=3, surface=surface)
# node = mesh.entity('node')
# fname = f"sphere_qtest.vtu"
# mesh.to_vtk(fname=fname)

qmesh = QuadrangleMesh.from_box(nx=1, ny=1)
# qmesh.uniform_refine(1)
node = qmesh.entity('node')
edge = qmesh.entity('edge')
cell = qmesh.entity('cell')
# print('node', node)
# print('edge', edge)
# print('cell', cell)

# fig = plt.figure()
# axes = fig.gca()
# qmesh.add_plot(axes)
# qmesh.find_node(axes, showindex=True)
# qmesh.find_edge(axes, showindex=True)
# qmesh.find_cell(axes, showindex=True)
# plt.show()

mesh = LagrangeQuadrangleMesh.from_box([0, 1, 0, 1], p=1, nx=1, ny=1)
mesh.uniform_refine(1)
node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')

fig = plt.figure()
ax = fig.add_subplot(111)
high_order_meshploter(ax, mesh)
plot_nodes(ax, node, fontsize=10, color='r')
plt.show()
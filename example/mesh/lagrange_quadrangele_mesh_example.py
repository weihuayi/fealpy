import matplotlib.pyplot as plt

from fealpy.backend import bm
from fealpy.geometry.implicit_surface import SphereSurface
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

surface = SphereSurface()
# mesh = QuadrangleMesh.from_unit_sphere_surface()
# lmesh = LagrangeQuadrangleMesh.from_quadrangle_mesh(mesh, p=2, surface=surface)
# fname = f"sphere_qtest.vtu"
# lmesh.to_vtk(fname=fname)

mesh = LagrangeQuadrangleMesh.from_box([0, 1, 0, 1], p=3, nx=2, ny=2)
#mesh.uniform_refine(1)
node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')


fig = plt.figure()
ax = fig.add_subplot(111)
high_order_meshploter(ax, mesh)
plot_nodes(ax, node, fontsize=10, color='r')
plt.show()




import matplotlib.pyplot as plt

from fealpy.backend import bm
from fealpy.geometry.implicit_curve import CircleCurve
from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh
from fealpy.mmesh.tool import high_order_meshploter

bm.set_backend('numpy')


def plot_nodes(ax, node, fontsize=10, color='k'):
    """
    """
    ax.scatter(node[:, 0], node[:, 1], color="blue", s=30, zorder=2)

    # 标记节点的行号
    for i, (x, y) in enumerate(node):
        ax.text(x, y, str(i), fontsize=fontsize, color=color)

def print_array(arr):
    """
    Print array with row indices.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (N, d).
    """
    for i, row in enumerate(arr):
        print(f"{i}: {row}")

curve = CircleCurve(center=(0.0, 0.0), radius=1.0)
mesh = TriangleMesh.from_one_hexagon()
mesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, 2, boundary=curve)

edge = mesh.entity('edge')

A = bm.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0]], dtype=bm.float64)

w = bm.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=bm.float64)
w = bm.stack((w @ A[[0, 4, 5]], w @ A[[1, 3, 5]], w @ A[[2, 4, 3]], w @ A[[3, 4, 5]]), axis=0)
cc = mesh.bc_to_point(w).reshape(-1, 2)
#ipoints = mesh.interpolation_points(4)

mesh.uniform_refine()

node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')
edge2cell = mesh.edge_to_cell()
cell2edge = mesh.cell_to_edge()

print_array(edge)
print_array(cell)
print_array(edge2cell)
print_array(cell2edge)

bc = bm.array([0.5, 0.5])
node1 = mesh.bc_to_point(bc)


fig = plt.figure()
ax = fig.add_subplot(111)
high_order_meshploter(ax, mesh)
plot_nodes(ax, node, fontsize=10, color='r')
plot_nodes(ax, node1, fontsize=15, color='k')
plot_nodes(ax, cc, fontsize=20, color='g')
plt.show()


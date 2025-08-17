
import matplotlib.pyplot as plt

from fealpy.backend import bm
from fealpy.geometry.implicit_curve import CircleCurve
from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh
from fealpy.mmesh.tool import high_order_meshploter

bm.set_backend('numpy')

curve = CircleCurve(center=(0.0, 0.0), radius=1.0)
mesh = TriangleMesh.from_unit_circle_gmsh(0.2)

mesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, 3)

node = mesh.entity('node')
isBdNode = mesh.boundary_node_flag()

bdNode, _ = curve.project(node[isBdNode, :])

node = bm.set_at(node, isBdNode, bdNode)



fig = plt.figure()
ax = fig.add_subplot(111)
high_order_meshploter(ax, mesh)
ax.plot(ipoints[:, 0], ipoints[:, 1], 'k.', markersize=10)
plt.show()


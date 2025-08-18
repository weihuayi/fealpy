
import matplotlib.pyplot as plt

from fealpy.backend import bm
from fealpy.geometry.implicit_curve import CircleCurve
from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh
from fealpy.mmesh.tool import high_order_meshploter

bm.set_backend('numpy')

curve = CircleCurve(center=(0.0, 0.0), radius=1.0)
mesh = TriangleMesh.from_one_hexagon()
mesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, 2, boundary=curve)

edge = mesh.entity('edge')

print(edge)

#ipoints = mesh.interpolation_points(4)

node = mesh.uniform_refine()


fig = plt.figure()
ax = fig.add_subplot(111)
high_order_meshploter(ax, mesh)
#ax.plot(ipoints[:, 0], ipoints[:, 1], 'k.', markersize=10)
ax.plot(node[:, 0], node[:, 1], 'k.', markersize=5)
plt.show()


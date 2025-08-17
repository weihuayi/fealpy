
import matplotlib.pyplot as plt

from fealpy.backend import bm
from fealpy.mesh import LagrangeTriangleMesh
from fealpy.mmesh.tool import high_order_meshploter

bm.set_backend('numpy')

mesh = LagrangeTriangleMesh.from_box([0, 1, 0, 1], 2, nx=2, ny=2)

print(mesh.linearmesh)

fig = plt.figure()
ax = fig.add_subplot(111)
high_order_meshploter(ax, mesh)
plt.show()


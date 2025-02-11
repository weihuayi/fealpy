from fealpy.mesh import IntervalMesh
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
interval = bm.array([0,1])
n = 2
mesh = IntervalMesh.from_interval_domain(interval , n)
mesh.uniform_refine()
print(mesh.entity('node'))

# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# plt.show()
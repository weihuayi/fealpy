from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('numpy')
# bm.set_backend('pytorch')
# bm.set_backend('jax')

from fealpy.experimental.mesh import UniformMesh2d

extent = [0, 2, 0, 2]
h = [1.5, 1]
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)

ip2 = mesh.interpolation_points(p=2)
edge = mesh.entity('edge')
cell = mesh.entity('cell')
edge2ipoint = mesh.edge_to_ipoint(p=2)
cell2ipoint = mesh.cell_to_ipoint(p=2)
edgenorm = mesh.edge_normal()
edgeunitnorrm = mesh.edge_unit_normal()
cellnorm = mesh.cell_normal()


import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
# mesh.find_node(axes, node=ip2, showindex=True)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()
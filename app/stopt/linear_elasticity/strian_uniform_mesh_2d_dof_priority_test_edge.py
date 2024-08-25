from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('numpy')
# bm.set_backend('pytorch')
# bm.set_backend('jax')

from fealpy.experimental.mesh import UniformMesh2d

extent = [0, 2, 0, 2]
h = [1.5, 1]
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)
# mesh.ipoints_ordering = 'yx'
mesh.ipoints_ordering = 'nec'

p = 2
ip = mesh.interpolation_points(p=p)
node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')
edgenorm = mesh.edge_normal()
edgeunitnorm = mesh.edge_unit_normal()
isBdNode = mesh.boundary_node_flag()
isBdEdge = mesh.boundary_edge_flag()
node2ipoint = mesh.node_to_ipoint(p=p, index=isBdNode)
edge2ipoint = mesh.edge_to_ipoint(p=p, index=isBdEdge)


import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ip, showindex=True)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
plt.show()
print("hhhhhhhhhhhhhhh")
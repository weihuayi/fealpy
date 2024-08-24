from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('numpy')
# bm.set_backend('pytorch')
# bm.set_backend('jax')

from fealpy.experimental.mesh import UniformMesh3d


extent = [0, 2, 0, 2, 0, 2]
h = [1.5, 1, 2]
origin = [0, 0, 0]
mesh = UniformMesh3d(extent, h, origin)
edge = mesh.entity('edge')
face = mesh.entity('face')
cell = mesh.entity('cell')
ip2 = mesh.interpolation_points(p=2)
# edgenorm = mesh.edge_normal()
# edgeunitnorrm = mesh.edge_unit_normal()
facenorm = mesh.face_normal()
faceunitnorrm = mesh.face_unit_normal()
cellnorm = mesh.cell_normal()


import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
# mesh.find_node(axes, node=ip2, showindex=True)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_face(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()
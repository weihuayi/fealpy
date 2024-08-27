from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('numpy')
# bm.set_backend('pytorch')
# bm.set_backend('jax')

from fealpy.experimental.mesh import UniformMesh3d


extent = [0, 2, 0, 2, 0, 2]
h = [1.5, 1, 2]

# extent = [0, 2, 0, 2, 0, 2]
# h = [1, 1, 1]
origin = [0, 0, 0]
mesh = UniformMesh3d(extent, h, origin)
# mesh.ipoints_ordering = 'zyx'
mesh.ipoints_ordering = 'nefc'

edge = mesh.entity('edge')
face = mesh.entity('face')
cell = mesh.entity('cell')

# facenorm = mesh.face_normal()
p = 3
ip2 = mesh.interpolation_points(p=p)
isBdNode = mesh.boundary_node_flag()
isBdEdge = mesh.boundary_edge_flag()
isBdFace = mesh.boundary_face_flag()
node2ipoint = mesh.node_to_ipoint(p=p, index=isBdNode)
edge2ipoint = mesh.edge_to_ipoint(p=p, index=isBdEdge)
face2ipoint = mesh.face_to_ipoint(p=p, index=isBdFace)
cell2ipoint = mesh.cell_to_ipoint(p=p)

facenorm = mesh.face_normal()
faceunitnorm = mesh.face_unit_normal()
# faceunitnorrm = mesh.face_unit_normal()
# cellnorm = mesh.cell_normal()


import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
mesh.find_node(axes, node=ip2, showindex=True)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_face(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
plt.show()
print("hhhhhhhhhhhhhhh")
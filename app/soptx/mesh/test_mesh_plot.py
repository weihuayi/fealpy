from fealpy.mesh import HexahedronMesh, QuadrangleMesh

nx, ny, nz = 2, 2, 2
domain_hex = [0, 1, 0, 1, 0, 1]
mesh_hex = HexahedronMesh.from_box(box=domain_hex, nx=nx, ny=ny, nz=nz)

# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh_hex.add_plot(axes)
# mesh_hex.find_node(axes, showindex=True)
# mesh_hex.find_cell(axes, showindex=True)
# plt.show()

domain_quad = [0, 1, 0, 1]
mesh_quad = QuadrangleMesh.from_box(box=domain_quad, nx=nx, ny=ny)

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh_quad.add_plot(axes)
mesh_quad.find_node(axes, showindex=True)
mesh_quad.find_edge(axes, showindex=True)
mesh_quad.find_cell(axes, showindex=True)
plt.show()
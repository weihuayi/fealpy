from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d

nx, ny = 1, 1
extent = [0, nx, 0, ny]
h = [1.0, 1.0]
origin = [0.0, 0.0]

mesh = UniformMesh2d(extent=extent, h=h, origin=origin)
ip2 = mesh.interpolation_points(p=4)
nip2 = mesh.number_of_local_ipoints(p=4)

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
mesh.find_node(axes, node=ip2, showindex=True)
plt.show()
print("----------------")

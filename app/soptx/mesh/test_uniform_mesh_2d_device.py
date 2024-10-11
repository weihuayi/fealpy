from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d

bm.set_backend('pytorch')
nx, ny = 1, 1
extent = [0, nx, 0, ny]
h = [1.0, 1.0]
origin = [0.0, 0.0]

mesh = UniformMesh2d(extent=extent, h=h, origin=origin, 
                    ipoints_ordering='yx', flip_direction='y',
                    device='cpu')


import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
plt.show()
print("----------------")

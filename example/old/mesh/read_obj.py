import numpy as np
import matplotlib.pyplot as plt


from fealpy.mesh import TriangleMesh

fname = '/home/why/geodataset/Mechanical/boat.obj'
mesh = TriangleMesh.from_obj_file(fname)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
plt.show()


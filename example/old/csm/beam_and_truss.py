import sys
import matplotlib.pyplot as plt
from fealpy.mesh import EdgeMesh


fname = sys.argv[1]

mesh = EdgeMesh.from_inp_file(fname)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
plt.show()



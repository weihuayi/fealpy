
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh

from fealpy_extent import generate_surface_mesh


node, cell = generate_surface_mesh()
mesh = TriangleMesh(node, cell)

fig = plt.figure()
#axes = fig.add_subplot(111, projection='3d')
axes = fig.gca(projection='3d')
mesh.add_plot(axes)
plt.show()


import ipdb
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMSystem 

csys = OCAMSystem.from_data()
plotter = OpenGLPlotter()

#csys.test_plain_domain(plotter, z=20)
#uv = csys.test_half_sphere_surface(plotter)

mesh, uv = csys.test_half_sphere_surface_with_cutting(plotter, ptype='O')

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
#plt.scatter(uv[:, 0], uv[:, 1])
plt.show()

plotter.run()


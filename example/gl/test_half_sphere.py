
import ipdb
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMSystem 

csys = OCAMSystem.from_data()
plotter = OpenGLPlotter()

#csys.test_plain_domain(plotter, z=20)
uv = csys.test_half_sphere_domain(plotter)


plt.scatter(uv[:, 0], uv[:, 1])
plt.show()

plotter.run()



import numpy as np

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMModel, OCAMSystem 

csys = OCAMSystem.from_data()
csys.show_images()

plotter = OpenGLPlotter()
csys.ellipsoid_mesh(plotter)
csys.sphere_mesh(plotter)
plotter.run()
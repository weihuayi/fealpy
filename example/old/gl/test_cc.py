import numpy as np
from fealpy.plotter.gl import OCAMModel
from fealpy.plotter.gl import OCAMSystem
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMSystem

csys = OCAMSystem.from_data('~/data/')
plotter = OpenGLPlotter()

csys.show_ground_mesh(plotter)
plotter.run()

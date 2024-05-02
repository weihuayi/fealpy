
import ipdb
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMSystem 

csys = OCAMSystem.from_data()
plotter = OpenGLPlotter()

csys.test_1(plotter, z=20)
plotter.run()


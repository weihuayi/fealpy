import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMSystem, OCAMModel, OptimizeParameter

csys = OCAMSystem.from_data()

opt = OptimizeParameter(csys) 
opt.optimize()



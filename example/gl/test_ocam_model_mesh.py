import numpy as np
from fealpy.plotter.gl import OCAMModel
from fealpy.plotter.gl import OCAMSystem
import matplotlib.pyplot as plt

csys = OCAMSystem.from_data()
model = csys.cams[0]

mesh = model.distmeshing(fh=None)
fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()

mesh = model.gmeshing()
fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()


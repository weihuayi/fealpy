import numpy as np
from fealpy.plotter.gl import OCAMModel
from fealpy.plotter.gl import OCAMSystem
import matplotlib.pyplot as plt

csys = OCAMSystem.from_data()
for i in range(6):
    model = csys.cams[i]

    mesh = model.gmeshing()
    #fig, axes = plt.subplots()
    #mesh.add_plot(axes)
    #plt.show()


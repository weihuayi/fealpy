import numpy as np
from fealpy.plotter.gl import OCAMModel
from fealpy.plotter.gl import OCAMSystem
import matplotlib.pyplot as plt

hmin=40
hmax=100

def sizing_function(p,*args):
    fd = args[0]
    h = hmin + np.abs(fd(p))*0.1
    h[h>hmax]=hmax
    return h

csys = OCAMSystem.from_data()
model = csys.cams[0]

mesh = model.distmeshing(hmin=hmin,fh=sizing_function)
fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()

mesh = model.gmeshing()
fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()


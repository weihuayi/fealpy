import numpy as np
from fealpy.plotter.gl import OCAMModel
from fealpy.plotter.gl import OCAMSystem
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMSystem 

csys = OCAMSystem.from_data()
plotter = OpenGLPlotter()

csys.show_screen_mesh(plotter, 4)
plotter.run()
for i in range(6):
    model = csys.cams[i]
    model.show_camera_image_and_mesh(outname='cam%d.png' % i)
    #mesh = model.gmshing_new()

    #fig, axes = plt.subplots()
    #mesh.add_plot(axes)
    #plt.show()


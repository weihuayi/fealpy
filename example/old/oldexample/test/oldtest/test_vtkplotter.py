from fealpy.plotter import VTKPlotter
from fealpy.plotter.shapes import Sphere


actor = Sphere()
plotter = VTKPlotter(shape=(2, 2))
plotter.show(actor)

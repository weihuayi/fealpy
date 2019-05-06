from fealpy.plotter import VTKPlotter
from fealpy.plotter.shapes import Sphere

actor = Sphere()
plotter = VTKPlotter()
plotter.show(actor)

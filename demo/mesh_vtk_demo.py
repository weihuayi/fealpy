#!/usr/bin/env python3
#

import matplotlib.pyplot as plt
from fealpy.pde.poisson_2d import CosCosData
from fealpy.plotter import VTKPlotter
from fealpy.plotter import meshactor


def simulation(queue=None):
    pde = CosCosData()
    mesh = pde.init_mesh(n=3, meshtype='tri')
    if queue is not None:
        queue.put({'mesh': mesh})
    return mesh


#plotter = VTKPlotter(shape=(1, 1), interactive=False, simulation=simulation)
#plotter.run()

plotter = VTKPlotter(shape=(1, 1), interactive=True)
mesh = simulation()
actor = meshactor(mesh)
plotter.show(actor)

#fig = plt.figure()
#axes = fig.gca()
#mesh.add_plot(axes)
#plt.show()

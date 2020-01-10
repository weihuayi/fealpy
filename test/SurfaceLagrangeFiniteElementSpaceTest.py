#!/usr/bin/env python3
# 
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.functionspace import SurfaceLagrangeFiniteElementSpace
from fealpy.pde.surface_poisson_model_3d import HeartSurfacetData


class SurfaceLagrangeFiniteElementSpaceTest:
    def __init__(self):
        pass

    def mesh_scale_test(self, plot=True):
        scale = 10
        pde = HeartSurfacetData()
        surface = pde.domain()
        mesh = pde.init_mesh()
        space = SurfaceLagrangeFiniteElementSpace(mesh, surface, p=1, scale=scale)
        mesh = space.mesh
        if plot is True:
            fig = plt.figure()
            axes = Axes3D(fig)
            mesh.add_plot(axes, showaxis=True)
            plt.show()

test = SurfaceLagrangeFiniteElementSpaceTest()
test.mesh_scale_test()

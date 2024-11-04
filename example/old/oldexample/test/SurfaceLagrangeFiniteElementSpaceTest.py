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

    def grad_recovery_test(self, p=1, plot=False):
        from fealpy.pde.surface_poisson_model_3d import SphereSinSinSinData 
        pde = SphereSinSinSinData()
        surface = pde.domain()
        mesh = pde.init_mesh()
        for i in range(4):
            space = SurfaceLagrangeFiniteElementSpace(mesh, surface, p=p)
            uI = space.interpolation(pde.solution)
            rg = space.grad_recovery(uI)
            error0 = space.integralalg.L2_error(pde.solution, uI.value)
            error1 = space.integralalg.L2_error(pde.gradient, rg.value)

            def f(x):
                return np.sum(rg.value(x)**2, axis=-1)
            eta = space.integralalg.integral(f, celltype=True)

            mesh.uniform_refine(surface=surface)
            print(error1)




test = SurfaceLagrangeFiniteElementSpaceTest()
#test.mesh_scale_test()
test.grad_recovery_test()


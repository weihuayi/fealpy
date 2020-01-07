import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh import Tritree
from fealpy.functionspace import SurfaceLagrangeFiniteElementSpace
from fealpy.pde.surface_poisson_model_3d import SphereSinSinSinData
class TritreeTest:
    def __init__(self):
        self.pde = SphereSinSinSinData()
        self.surface = self.pde.domain()
        self.mesh = self.pde.init_mesh(1)
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        self.tritree = Tritree(node, cell)

    def test_adaptive(self):
        options = self.tritree.adaptive_options(maxrefine=1)
        mesh = self.tritree.to_conformmesh()
        for i in range(1):
            space = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p=1)
            uI = space.interpolation(self.pde.solution)
            phi = lambda x : uI.grad_value(x)**2
            eta = space.integralalg.integral(lambda x : uI.grad_value(x)**2, celltype=True, barycenter=True)
            eta = eta.sum(axis=-1)
            self.tritree.adaptive(eta, options, surface=self.surface)
            mesh = self.tritree.to_conformmesh()
            print(self.tritree.celldata['idxmap'])
            cell = self.tritree.entity('cell')
            print(cell[15])
            cell = mesh.entity('cell')
            print(cell[104])
            print(cell[140])

        fig = pl.figure()
        axes = a3.Axes3D(fig)
        self.tritree.add_plot(axes)
        plt.show()

    def test_interpolation(self):
        options = self.tritree.adaptive_options(maxrefine=1)
        mesh = self.tritree.to_conformmesh()
        space = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p=2)
        uI = space.interpolation(self.pde.solution)
        self.tritree.interpolation(uI)

        phi = lambda x : uI.grad_value(x)**2
        eta = space.integralalg.integral(lambda x : uI.grad_value(x)**2, celltype=True, barycenter=True)
        eta = eta.sum(axis=-1)
        self.tritree.adaptive(eta, options, surface=self.surface)
        mesh = self.tritree.to_conformmesh()

        fig = pl.figure()
        axes = a3.Axes3D(fig)
        self.tritree.add_plot(axes)
        plt.show()





test = TritreeTest()
test.test_adaptive()
#test.test_interpolation()

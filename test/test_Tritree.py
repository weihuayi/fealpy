import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh import Tritree
from fealpy.functionspace import SurfaceLagrangeFiniteElementSpace
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.pde.surface_poisson_model_3d import SphereSinSinSinData
from fealpy.pde.poisson_2d import CosCosData
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
        options = self.tritree.adaptive_options(maxrefine=1, p=2)
        mesh = self.tritree.to_conformmesh()
        space = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p=2)
        uI = space.interpolation(self.pde.solution)
        error0 = space.integralalg.L2_error(self.pde.solution, uI)
        print(error0)
        data = self.tritree.interpolation(uI)
        options['data'] = {'q':data}
        if 1:
            eta = space.integralalg.integral(lambda x : uI.grad_value(x)**2, celltype=True, barycenter=True)
            eta = eta.sum(axis=-1)
            self.tritree.adaptive(eta, options, surface=self.surface)
        else:
            self.tritree.uniform_refine(options=options, surface=self.surface)

        mesh = self.tritree.to_conformmesh(options)
        space = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p=2)
        data = options['data']['q']
        uh = space.to_function(data)

        error1 = space.integralalg.L2_error(self.pde.solution, uh)
        print(error1)

        if 1:
            fig = pl.figure()
            axes = a3.Axes3D(fig)
            self.tritree.add_plot(axes)
            plt.show()

    def test_interpolation_1(self):
        pde = CosCosData()
        mesh = pde.init_mesh(n=3)
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        tritree = Tritree(node, cell)

        options = tritree.adaptive_options(maxrefine=1, p=2)
        mesh = tritree.to_conformmesh()
        space = LagrangeFiniteElementSpace(mesh, p=2)
        uI = space.interpolation(pde.solution)
        error0 = space.integralalg.L2_error(pde.solution, uI)
        print(error0)
        data = tritree.interpolation(uI)
        options['data'] = {'q':data}
        if 1:
            eta = space.integralalg.integral(lambda x : uI.grad_value(x)**2, celltype=True, barycenter=True)
            eta = eta.sum(axis=-1)
            tritree.adaptive(eta, options)
        else:
            tritree.uniform_refine(options=options)

        mesh = tritree.to_conformmesh(options)
        space = LagrangeFiniteElementSpace(mesh, p=2)
        data = options['data']['q']
        uh = space.to_function(data)

        error1 = space.integralalg.L2_error(pde.solution, uh)
        print(error1)




test = TritreeTest()
#test.test_adaptive()
#test.test_interpolation()
test.test_interpolation_1()

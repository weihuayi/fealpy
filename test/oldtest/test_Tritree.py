import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import pylab as pl

from fealpy.mesh import Tritree, TriangleMesh
from fealpy.functionspace import SurfaceLagrangeFiniteElementSpace
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.pde.surface_poisson_model_3d import SphereSinSinSinData, HeartSurfacetData
from fealpy.pde.poisson_2d import CosCosData

class TritreeTest:
    def __init__(self):
        self.pde = SphereSinSinSinData()
        self.surface = self.pde.domain()
        self.mesh = self.pde.init_mesh(4)
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

    def test_interpolation_surface(self, p=1):
        options = self.tritree.adaptive_options(maxrefine=1, p=p)
        mesh = self.tritree.to_conformmesh()
        space = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p=p)
        uI = space.interpolation(self.pde.solution)
        print('1:', space.number_of_global_dofs())
        print('2:', uI.shape)
        error0 = space.integralalg.L2_error(self.pde.solution, uI)
        print(error0)
        data = self.tritree.interpolation(uI)
        print('3', data.shape)
        options['data'] = {'q':data}
        if 1:
            eta = space.integralalg.integral(lambda x : uI.grad_value(x)**2, celltype=True, barycenter=True)
            eta = eta.sum(axis=-1)
            self.tritree.adaptive(eta, options, surface=self.surface)
        else:
            self.tritree.uniform_refine(options=options, surface=self.surface)

        mesh = self.tritree.to_conformmesh(options)
        space = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p=p)
        data = options['data']['q']
        print('data:', data.shape)
        uh = space.to_function(data)
        print('uh:', uh.shape)
        error1 = space.integralalg.L2_error(self.pde.solution, uh)
        print(error1)

        uI = space.interpolation(self.pde.solution)
        error2 = space.integralalg.L2_error(self.pde.solution, uI)
        print(error2)

        data = self.tritree.interpolation(uI)
        options['data'] = {'q':data}
        if 0:
            eta = space.integralalg.integral(lambda x : uI.grad_value(x)**2, celltype=True, barycenter=True)
            eta = eta.sum(axis=-1)
            self.tritree.adaptive(eta, options, surface=self.surface)
        else:
            self.tritree.uniform_refine(options=options, surface=self.surface)

        mesh = self.tritree.to_conformmesh(options)
        space = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p=p)
        data = options['data']['q']
        uh = space.to_function(data)
        error3 = space.integralalg.L2_error(self.pde.solution, uh)
        print(error3)

        if 0:
            fig = pl.figure()
            axes = a3.Axes3D(fig)
            self.tritree.add_plot(axes)
            plt.show()
        else:
            fig = pl.figure()
            axes = a3.Axes3D(fig) 
            mesh.add_plot(axes)
            plt.show()

    def test_interpolation_plane(self):

        def u(p):
            x = p[..., 0]
            y = p[..., 1]
            return x*y

        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)

        node = mesh.entity('node')
        cell = mesh.entity('cell')
        tritree = Tritree(node, cell)
        mesh = tritree.to_conformmesh()

        space = LagrangeFiniteElementSpace(mesh, p=2)
        uI = space.interpolation(u)
        error0 = space.integralalg.L2_error(u, uI)

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, node=space.interpolation_points(), showindex=True)

        data = tritree.interpolation(uI)
        options = tritree.adaptive_options(
                method='numrefine', data={"q":data},
                maxrefine=1, p=2)
        if 1:
            #eta = space.integralalg.integral(lambda x : uI.grad_value(x)**2, celltype=True, barycenter=True)
            #eta = eta.sum(axis=-1)
            eta = np.array([1, 0], dtype=np.int)
            tritree.adaptive(eta, options)
        else:
            tritree.uniform_refine(options=options)

        fig = plt.figure()
        axes = fig.gca()
        tritree.add_plot(axes)
        tritree.find_node(axes, showindex=True)

        mesh = tritree.to_conformmesh(options)
        space = LagrangeFiniteElementSpace(mesh, p=2)
        data = options['data']['q']
        uh = space.to_function(data)


        error1 = space.integralalg.L2_error(u, uh)

        data = tritree.interpolation(uh)
        isLeafCell = tritree.is_leaf_cell()

        fig = plt.figure()
        axes = fig.gca()
        tritree.add_plot(axes)
        tritree.find_node(axes, node=space.interpolation_points(), showindex=True)
        tritree.find_cell(axes, index=isLeafCell, showindex=True)

        options = tritree.adaptive_options(
                method='numrefine', data={"q":data},
                maxrefine=1, maxcoarsen=1, p=2)
        if 1:
            #eta = space.integralalg.integral(lambda x : uI.grad_value(x)**2, celltype=True, barycenter=True)
            #eta = eta.sum(axis=-1)
            eta = np.array([-1, -1, -1, -1, 0, 1], dtype=np.int)
            tritree.adaptive(eta, options)
        else:
            tritree.uniform_refine(options=options)

        mesh = tritree.to_conformmesh(options)
        space = LagrangeFiniteElementSpace(mesh, p=2)
        data = options['data']['q']
        uh = space.to_function(data)

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, node=space.interpolation_points(), showindex=True)
        mesh.find_cell(axes, showindex=True)

        error2 = space.integralalg.L2_error(u, uh)
        print(error0)
        print(error1)
        print(error2)
        plt.show()


class HeartTritreeTest:
    def __init__(self, scale=2):
        self.scale = scale
        self.pde = HeartSurfacetData()
        self.surface = self.pde.domain()
        self.mesh = self.pde.init_mesh()
        self.mesh.uniform_refine(surface=self.surface)
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        self.tritree = Tritree(node, cell)

    def test_interpolation_surface(self, p=1):
        options = self.tritree.adaptive_options(maxrefine=1, p=p)
        mesh = self.tritree.to_conformmesh()

        fig = pl.figure()
        axes = a3.Axes3D(fig)
        mesh.add_plot(axes, showaxis=True)

        space = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p=p, scale=self.scale)
        uI = space.interpolation(self.pde.solution)
        print('1:', space.number_of_global_dofs())
        print('2:', uI.shape)
        error0 = space.integralalg.L2_error(self.pde.solution, uI)
        print(error0)
        data = self.tritree.interpolation(uI)
        print('3', data.shape)
        options['data'] = {'q':data}
        if 0:
            eta = space.integralalg.integral(lambda x : uI.grad_value(x)**2, celltype=True, barycenter=True)
            eta = eta.sum(axis=-1)
            self.tritree.adaptive(eta, options, surface=self.surface)
        else:
            self.tritree.uniform_refine(options=options, surface=self.surface)

        fig = pl.figure()
        axes = a3.Axes3D(fig)
        self.tritree.add_plot(axes, showaxis=True)

        mesh = self.tritree.to_conformmesh(options)

        fig = pl.figure()
        axes = a3.Axes3D(fig)
        mesh.add_plot(axes, showaxis=True)

        space = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p=p, scale=self.scale)
        mesh0 = space.mesh

        fig = pl.figure()
        axes = a3.Axes3D(fig)
        mesh0.add_plot(axes, showaxis=True)

        data = options['data']['q']
        print('data:', data.shape)
        uh = space.to_function(data)
        print('uh:', uh.shape)
        error1 = space.integralalg.L2_error(self.pde.solution, uh)
        print(error1)

        uI = space.interpolation(self.pde.solution)
        error2 = space.integralalg.L2_error(self.pde.solution, uI)
        print(error2)

        if 0:
            fig = pl.figure()
            axes = a3.Axes3D(fig)
            self.tritree.add_plot(axes)
            plt.show()
        else:
            fig = pl.figure()
            axes = a3.Axes3D(fig)
            mesh.add_plot(axes, showaxis=True)
            plt.show()




test = HeartTritreeTest()
#test.test_adaptive()
#test.test_interpolation_plane()
test.test_interpolation_surface(p=2)

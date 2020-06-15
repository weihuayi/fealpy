#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh, MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace.femdof import multi_index_matrix2d



class RaviartThomasFiniteElementSpace2dTest:

    def __init__(self):
        self.meshfactory = MeshFactory()

    def show_basis(self, p=0):
        h = 0.5
        box = [-h, 1+h, -h, 1+h]

        node = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]], dtype=np.float)
        cell = np.array([[1, 2, 0]], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        space = RaviartThomasFiniteElementSpace2d(mesh, p=p)
        fig = plt.figure()
        space.show_basis(fig, box=box)
        print(space.bcoefs)

        cell = np.array([[3, 0, 2]], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        space = RaviartThomasFiniteElementSpace2d(mesh, p=p)
        fig = plt.figure()
        space.show_basis(fig, box=box)
        print(space.bcoefs)

        plt.show()

    def interpolation(self, n=0, p=0, plot=True):
        
        box = [-0.5, 1.5, -0.5, 1.5]
        pde = CosCosData()
        mesh = pde.init_mesh(n=n, meshtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=p)
        uI = space.interpolation(pde.flux)
        error = space.integralalg.L2_error(pde.flux, uI)
        print(error)
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes, box=box)
            plt.show()

    def solve_poisson_2d(self, n=3, p=0):
        pde = CosCosData()
        mesh = pde.init_mesh(n=n, meshtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=p)

        gdof = space.number_of_global_dofs()


        vh = space.function()
        uh = space.smspace.function()

        A = space.mass_matrix()
        B = space.div_matrix()
        F0 = space.neumann_boundary_vector(pde.dirichlet)
        F1 = space.source_vector(pde.source)

        AA = bmat([[A, -B], [-B.T, None]], format='csr')
        FF = np.r_['0', F0, F1]

        x = spsolve(AA, FF)

        vh[:] = x[:gdof]
        uh[:] = x[gdof:]

        error = space.integralalg.L2_error(pde.flux, vh)
        print(error)



test = RaviartThomasFiniteElementSpace2dTest()

if sys.argv[1] == "show_basis":
    p = int(sys.argv[2])
    test.show_basis(p=p)
elif sys.argv[1] == "interpolation":
    n = int(sys.argv[2])
    p = int(sys.argv[3])
    test.interpolation(n=n, p=p)
elif sys.argv[1] == "solve_poisson_2d":
    n = int(sys.argv[2])
    p = int(sys.argv[3])
    test.solve_poisson_2d(n=n, p=p)


    

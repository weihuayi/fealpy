#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh, MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.pde.poisson_2d import CosCosData, X2Y2Data
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

    def edge_basis(self, plot=True):
        pde = X2Y2Data()
        mesh = pde.init_mesh(n=0, meshtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=0)
        qf = mesh.integrator(3, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        index = mesh.ds.boundary_edge_index()
        ps = mesh.bc_to_point(bcs, etype='edge', index=index)
        phi = space.edge_basis(bcs, index=index) 

        if plot:
            box = [-0.5, 1.5, -0.5, 1.5]
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes, box=box)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            node = ps.reshape(-1, 2)
            uv = phi.reshape(-1, 2)
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1])
            plt.show()

    def set_dirichlet_bc(self, plot=True):
        pass

    def solve_poisson_2d(self, n=3, p=0, plot=True):
        pde = CosCosData()

        mesh = pde.init_mesh(n=n, meshtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=p)

        udof = space.number_of_global_dofs()
        pdof = space.smspace.number_of_global_dofs()
        gdof = udof + pdof

        uh = space.function()
        ph = space.smspace.function()
        A = space.stiff_matrix()
        B = space.div_matrix()
        F1 = space.smspace.source_vector(pde.source)
        AA = bmat([[A, -B], [-B.T, None]], format='csr')

        if True:
            F0 = -space.set_neumann_bc(pde.dirichlet)
            FF = np.r_['0', F0, F1]
            x = spsolve(AA, FF).reshape(-1)
            uh[:] = x[:udof]
            ph[:] = x[udof:]
            error0 = space.integralalg.L2_error(pde.flux, uh)

            def f(bc):
                xx = mesh.bc_to_point(bc)
                return (pde.solution(xx) - ph(xx))**2
            error1 = space.integralalg.integral(f)
            print(error0, error1)
        else:
            isBdDof = -space.set_dirichlet_bc(uh, pde.neumann)
            x = np.r_['0', uh, ph] 
            isBdDof = np.r_['0', isBdDof, np.zeros(pdof, dtype=np.bool_)]
            
            FF = np.r_['0', np.zeros(udof, dtype=np.float64), F1]

            FF -= AA@x
            bdIdx = np.zeros(gdof, dtype=np.int)
            bdIdx[isBdDof] = 1
            Tbd = spdiags(bdIdx, 0, gdof, gdof)
            T = spdiags(1-bdIdx, 0, gdof, gdof)
            AA = T@AA@T + Tbd
            FF[isBdDof] = x[isBdDof]
            x[:] = spsolve(AA, FF)
            uh[:] = x[:udof]
            ph[:] = x[udof:]

            error0 = space.integralalg.L2_error(pde.flux, uh)

            def f(bc):
                xx = mesh.bc_to_point(bc)
                return (pde.solution(xx) - ph(xx))**2
            error1 = space.integralalg.integral(f)
            print(error0, error1)

        if plot:
            box = [-0.5, 1.5, -0.5, 1.5]
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes, box=box)
            #mesh.find_node(axes, showindex=True)
            #mesh.find_edge(axes, showindex=True)
            #mesh.find_cell(axes, showindex=True)
            node = ps.reshape(-1, 2)
            uv = uh.reshape(-1, 2)
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1])
            plt.show()

    def sympy_compute(self, plot=True):
        import sympy as sp
        from sympy.abc import x, y, z

        if plot:
            pde = CosCosData()
            mesh = pde.init_mesh(n=0, meshtype='tri')
            n = mesh.edge_unit_normal()
            print("n:\n", n)
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()



test = RaviartThomasFiniteElementSpace2dTest()

if sys.argv[1] == "show_basis":
    p = int(sys.argv[2])
    test.show_basis(p=p)
elif sys.argv[1] == "sympy_compute":
    test.sympy_compute()
elif sys.argv[1] == "interpolation":
    n = int(sys.argv[2])
    p = int(sys.argv[3])
    test.interpolation(n=n, p=p)
elif sys.argv[1] == "solve_poisson_2d":
    n = int(sys.argv[2])
    p = int(sys.argv[3])
    test.solve_poisson_2d(n=n, p=p, plot=False)
elif sys.argv[1] == 'edge_basis':
    test.edge_basis()



    

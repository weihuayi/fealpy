#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.pde.timeharmonic_2d import CosSinData
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d 
from fealpy.functionspace.femdof import multi_index_matrix2d



class FirstKindNedelecFiniteElementSpace2dTest:
    def __init__(self):
        pass

    def show_basis(self):
        h = 0.5
        box = [-h, 1+h, -h, np.sqrt(3)/2+h]
        mesh = MeshFactory.one_triangle_mesh()
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=0)
        fig = plt.figure()
        space.show_basis(fig, box=box)
        plt.show()

    def interpolation(self, n=4, p=0, plot=True):
        
        box = [-0.5, 1.5, -0.5, 1.5]

        def u(p):
            x = p[..., 0]
            y = p[..., 1]
            val = np.zeros_like(p)
            pi = np.pi
            val[..., 0] = np.sin(pi*x)*np.cos(pi*y)
            val[..., 1] = np.sin(pi*x)*np.cos(pi*y)
            return val

        mesh = MeshFactory.boxmesh2d([0, 1, 0, 1], nx=n, ny=n, meshtype='tri')
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=p)
        uI = space.interpolation(u)
        error = space.integralalg.L2_error(u, uI)
        print(error)
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes, box=box)
            plt.show()

    def solve_time_harmonic_2d(self, n=3, p=0, plot=True):
        pde = CosSinData()
        mesh = pde.init_mesh(n=n, meshtype='tri')
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=p)

        gdof = space.number_of_global_dofs()
        uh = space.function()

        A = space.curl_matrix() - space.mass_matrix()
        F = space.source_vector(pde.source)

        isBdDof = space.boundary_dof()
        bdIdx = np.zeros(gdof, dtype=np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, gdof, gdof)
        T = spdiags(1-bdIdx, 0, gdof, gdof)
        A = T@A@T + Tbd
        F[isBdDof] = 0 
        uh[:] = spsolve(A, F)


        error0 = space.integralalg.L2_error(pde.solution, uh)
        error1 = space.integralalg.L2_error(pde.curl, uh.curl_value)
        print(error0, error1)

        if plot:
            box = [-0.5, 1.5, -0.5, 1.5]
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes, box=box)
            #mesh.find_node(axes, showindex=True)
            #mesh.find_edge(axes, showindex=True)
            #mesh.find_cell(axes, showindex=True)
            #node = ps.reshape(-1, 2)
            #uv = phi.reshape(-1, 2)
            #axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1])
            plt.show()

test = FirstKindNedelecFiniteElementSpace2dTest()
if sys.argv[1] == "show_basis":
    test.show_basis()
elif sys.argv[1] == "interpolation":
    n = int(sys.argv[2])
    p = int(sys.argv[3])
    test.interpolation(n=n, p=p)
elif sys.argv[1] == "solve":
    n = int(sys.argv[2])
    p = int(sys.argv[3])
    test.solve_time_harmonic_2d(n=n, p=p)

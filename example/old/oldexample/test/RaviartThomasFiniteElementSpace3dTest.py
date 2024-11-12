#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_3d import CosCosCosData, X2Y2Z2Data
from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace3d
from fealpy.functionspace.femdof import multi_index_matrix2d



class RaviartThomasFiniteElementSpace3dTest:

    def __init__(self):
        self.meshfactory = MeshFactory()

    def show_basis(self, p=0):
        mesh = self.meshfactory.one_tetrahedron_mesh(ttype='equ')
        space = RaviartThomasFiniteElementSpace3d(mesh, p=p, q=2)
        fig = plt.figure()
        space.show_basis(fig)
        plt.show()

    def basis_coefficients(self, n=0, p=0):
        pde = X2Y2Z2Data()
        mesh = pde.init_mesh(n=n, meshtype='tet')
        space = RaviartThomasFiniteElementSpace3d(mesh, p=p)
        C = space.basis_coefficients()

        return C

    def solve_poisson_3d(self, n=0, p=0, plot=True):
        pde = CosCosCosData()
        mesh = pde.init_mesh(n=n, meshtype='tet')
        space = RaviartThomasFiniteElementSpace3d(mesh, p=p, q=p+4)

        udof = space.number_of_global_dofs()
        pdof = space.smspace.number_of_global_dofs()
        gdof = udof + pdof

        uh = space.function()
        ph = space.smspace.function()
        A = space.stiff_matrix()
        B = space.div_matrix()
        F1 = space.source_vector(pde.source)
        AA = bmat([[A, -B], [-B.T, None]], format='csr')

        F0 = space.set_neumann_bc(pde.dirichlet)
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



test = RaviartThomasFiniteElementSpace3dTest()

if sys.argv[1] == "show_basis":
    p = int(sys.argv[2])
    test.show_basis(p=p)
if sys.argv[1] == "basis_coef":
    n = int(sys.argv[2])
    p = int(sys.argv[3])
    test.basis_coefficients(n=n, p=p)
if sys.argv[1] == "solve_poisson_3d":
    n = int(sys.argv[2])
    p = int(sys.argv[3])
    test.solve_poisson_3d(n=n, p=p)


#!/usr/bin/env python3
# 
import sys
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace.femdof import multi_index_matrix2d



class RaviartThomasFiniteElementSpace2dTest:

    def __init__(self):
        self.meshfactory = MeshFactory()

    def show_basis(self, p=0):
        h = 0.5
        box = [-h, 1+h, -h, np.sqrt(3)/2+h]
        mesh = self.meshfactory.one_triangle_mesh('equ')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=p)
        fig = plt.figure()
        space.show_basis(fig, box=box)
        plt.show()

    def interpolation(self, n=3, plot=True):
        pde = CosCosData()
        mesh = pde.init_mesh(n=n, meshtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=0)
        uI = space.interpolation(pde.flux)
        error = space.integralalg.L2_error(pde.flux, uI)
        print(error)

    def solve_poisson_2d(self):

        pde = CosCosData()
        mesh = pde.init_mesh(n=3, methtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=0)
        A = space.mass_matrix()
        B = space.div_matrix()



test = RaviartThomasFiniteElementSpace2dTest()

if sys.argv[1] == "show_basis":
    p = int(sys.argv[2])
    test.show_basis(p=p)
elif sys.argv[1] == "interpolation":
    n = int(sys.argv[2])
    test.interpolation(n=n)

    

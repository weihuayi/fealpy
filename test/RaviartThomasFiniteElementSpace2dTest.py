#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace.femdof import multi_index_matrix2d



class RaviartThomasFiniteElementSpace2dTest:

    def __init__(self):
        self.meshfactory = MeshFactory()

    def show_basis(self):
        h = 0.5
        box = [-h, 1+h, -h, np.sqrt(3)/2+h]
        mesh = self.meshfactory.one_triangle_mesh('equ')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=2, q=2)
        fig = plt.figure()
        space.show_basis(fig, box=box)
        plt.show()

    def interpolation(self):
        pde = CosCosData()
        mesh = pde.init_mesh(n=3, methtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=0)

    def solve_poisson_2d(self):

        pde = CosCosData()
        mesh = pde.init_mesh(n=3, methtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=0)
        A = space.mass_matrix()
        B = space.div_matrix()



test = RaviartThomasFiniteElementSpace2dTest()


if sys.argv[1] == "show_basis":
    test.show_basis_test()

    

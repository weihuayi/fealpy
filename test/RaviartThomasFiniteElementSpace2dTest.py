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

    def show_basis_test(self):
        h = 0.5
        box = [-h, 1+h, -h, np.sqrt(3)/2+h]
        mesh = self.meshfactory.one_triangle_mesh('equ')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=2, q=2)
        fig = plt.figure()
        space.show_basis(fig, box=box)
        plt.show()


test = RaviartThomasFiniteElementSpace2dTest()
test.show_basis_test()

    

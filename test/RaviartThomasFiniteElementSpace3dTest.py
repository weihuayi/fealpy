#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace3d
from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace.femdof import multi_index_matrix2d



class RaviartThomasFiniteElementSpace3dTest:

    def __init__(self):
        self.meshfactory = MeshFactory()

    def show_basis_test(self, p=0):
        mesh = self.meshfactory.one_tetrahedron_mesh(ttype='equ')
        space = RaviartThomasFiniteElementSpace3d(mesh, p=p, q=2)
        fig = plt.figure()
        space.show_basis(fig)
        plt.show()


test = RaviartThomasFiniteElementSpace3dTest()
test.show_basis_test(p=0)

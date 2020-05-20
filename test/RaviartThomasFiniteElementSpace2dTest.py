#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace.femdof import multi_index_matrix2d



class RaviartThomasFiniteElementSpace2dTest:

    def __init__(self):
        pass

    def space_test(self):
        pde = CosCosData()
        mesh = pde.init_mesh(n=0, meshtype='tri')
        space = RaviartThomasFiniteElementSpace2d(mesh, p=2, q=2)
        bcs = multi_index_matrix2d(3)/3
        ps = mesh.bc_to_point(bcs)
        phi = space.basis(bcs)
        print(phi.shape)

        if True:
            fig = plt.figure()

            axes = fig.add_subplot(1, 2, 1)
            mesh.add_plot(axes)
            node = ps[:, 0, :]
            uv = phi[:, 0, 1, :]
            mesh.find_node(axes, node=node, showindex=True)
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1], units='xy')

            axes = fig.add_subplot(1, 2, 2)
            mesh.add_plot(axes)
            node = ps[:, 1, :]
            uv = phi[:, 1, 1, :]
            mesh.find_node(axes, node=ps[:, 1, :], showindex=True)
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1], units='xy')
            plt.show()


test = RaviartThomasFiniteElementSpace2dTest()
test.space_test()

    

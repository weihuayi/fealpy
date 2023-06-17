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

class FractureRTTest:
    def __init__(self):
        self.mf = MeshFactory()

    def mesh_with_fracture(self, plot=True):

        box = [0, 1, 0, 1]
        mesh = self.mf.boxmesh2d(box, nx=10, ny=10, meshtype='tri')

        def is_fracture(p):
            x = p[..., 0]
            y = p[..., 1]
            flag0 = (x == 0.5) & ( y > 0.2) & (y < 0.8)
            flag1 = (y == 0.5) & ( x > 0.2) & (x < 0.8)
            return flag0 | flag1

        bc = mesh.entity_barycenter('edge')
        isFEdge = is_fracture(bc)
        mesh.edgedata['fracture'] = isFEdge
        NE = mesh.number_of_edges()
        print('边的个数：', NE)

        space = RaviartThomasFiniteElementSpace2d(mesh, p=0)
        gdof = space.number_of_global_dofs()
        print('gdof:', gdof)

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_edge(axes, index=isFEdge, color='r')
            plt.show()




test = FractureRTTest()

if sys.argv[1] == 'fracture':
    test.mesh_with_fracture()


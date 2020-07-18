#!/usr/bin/env python3
# 
import sys
import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from fealpy.writer import MeshWriter
from fealpy.mesh import HalfEdgeMesh2d
from fealpy.mesh.TriAdvancingFrontAlg import TriAdvancingFrontAlg

class TriAdvancingFrontAlgTest():

    def __init__(self):
        pass


    def square_domain(self, plot=True):

        node = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float)
        edge = np.array([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int)
        edge2subdomain = np.array([(1, 0), (1, 0), (1, 0), (1, 0)],
                dtype=np.int)

        mesh = HalfEdgeMesh2d.from_edges(node, edge, edge2subdomain)
        alg = TriAdvancingFrontAlg(mesh)
        alg.run()
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes)
            plt.show()



test = TriAdvancingFrontAlgTest()
if sys.argv[1] == 'square':
    test.square_domain()

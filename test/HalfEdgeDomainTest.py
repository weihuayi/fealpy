#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeDomain

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree


class HalfEdgeDomainTest:

    def __init__(self):
        pass

    def square_domain_test(self, plot=True):

        node = np.array([
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
        halfedge = np.array([
            (1, 0, 1, 3, 4, 1, 1),
            (2, 0, 2, 0, 5, 1, 1),
            (3, 0, 3, 1, 6, 1, 1),
            (0, 0, 0, 2, 7, 1, 1),
            (0, 1, 7, 5, 0, 0, 1),
            (1, 1, 4, 6, 1, 0, 1),
            (2, 1, 5, 7, 2, 0, 1),
            (3, 1, 6, 4, 3, 0, 1)], dtype=np.int)

        domain = HalfEdgeDomain(node, halfedge, 1)
        domain.uniform_refine(n=4)
        
        np.random.seed(0)
        points = np.random.rand(10,2)
        vor = Voronoi(domain.node)
        print('region:', vor.regions)
        print('ridge_vertices:', vor.ridge_vertices)
        print('ridge_points:', vor.ridge_points)



        if plot:
            fig = voronoi_plot_2d(vor)
            axes = fig.gca()
            domain.add_plot(axes)
            domain.find_node(axes, node=vor.vertices, showindex=True)
            domain.find_node(axes, node=points, color='blue', showindex=True)
            axes.set_xlim([-0.1, 1.1])
            axes.set_ylim([-0.1, 1.1])
            plt.show()



test = HalfEdgeDomainTest()
test.square_domain_test()

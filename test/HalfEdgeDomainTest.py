#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeDomain, HalfEdgeMesh

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree


class HalfEdgeDomainTest:

    def __init__(self):
        pass

    def square_domain_test(self, plot=True):

        node = np.array([
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
        halfedge = np.array([
            (1, 0, 1, 3, 4, 1),
            (2, 0, 2, 0, 5, 1),
            (3, 0, 3, 1, 6, 1),
            (0, 0, 0, 2, 7, 1),
            (0, 1, 7, 5, 0, 0),
            (1, 1, 4, 6, 1, 0),
            (2, 1, 5, 7, 2, 0),
            (3, 1, 6, 4, 3, 0)], dtype=np.int)

        domain = HalfEdgeDomain(node, halfedge, NS=1)
        mesh = HalfEdgeMesh(node, halfedge, 1)

        np.random.seed(0)
        points = np.random.rand(10,2)
        vor = Voronoi(points)
        rp = vor.ridge_points
        rv = np.array(vor.ridge_vertices)
        isInfVertices = rv[:, 0] == -1
        print('ridge_points:\n', rp[isInfVertices])
        print('ridge_vertices:\n', rv[isInfVertices])

        mesh = domain.create_finite_voronoi(points)


        if plot:
            fig = plt.figure()
            axes = fig.gca()
            #voronoi_plot_2d(vor, ax=axes)
            mesh.add_plot(axes)
            mesh.find_node(axes, node=points, color='r', showindex=True)
            mesh.find_node(axes, node=vor.vertices, color='b', showindex=True)
            plt.show()



test = HalfEdgeDomainTest()
test.square_domain_test()

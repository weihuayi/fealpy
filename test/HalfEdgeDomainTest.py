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

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            #voronoi_plot_2d(vor, ax=axes)
            mesh.add_plot(axes)
            mesh.find_node(axes, node=points, color='r', showindex=True)
            mesh.find_node(axes, node=vor.vertices, color='b', showindex=True)
            plt.show()

    def advance_trimesh_test(self, plot=True):
        vertices = np.array([
            (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
            (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)
            ], dtype=np.float)
        halfedge = np.array([
            (0, 1,  1,  3,  8, 1), #0
            (1, 1,  2,  0,  9, 1), #1
            (2, 1,  3,  1, 10, 1), #2
            (3, 1,  0,  2, 11, 1), #3
            (4, 1,  5,  7, 12, 1), #4
            (7, 1,  6,  4, 13, 1), #5
            (6, 1,  7,  5, 14, 1), #6
            (5, 1,  4,  6, 15, 1), #7
            (3, 0, 11,  9,  0, 0), #8
            (0, 0,  8, 10,  1, 0), #9
            (1, 0,  9, 11,  2, 0), #10
            (2, 0, 10,  8,  3, 0), #11
            (5, -1, 15, 13,  4, 0), #12
            (4, -1, 12, 14,  5, 0), #13
            (7, -1, 13, 15,  6, 0), #14
            (6, -1, 14, 12,  7, 0)],#15
            dtype=np.int)

        domain = HalfEdgeDomain(vertices, halfedge)
        for i in range(4):
            isMarkedHEdge= (domain.halfedge[:, 1] == 0)
            domain.halfedge_adaptive_refine(isMarkedHEdge)
        
        for i in range(6):
            isMarkedHEdge= (domain.halfedge[:, 1] == -1)
            domain.halfedge_adaptive_refine(isMarkedHEdge)

        mesh = domain.to_halfedgemesh()
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, markersize=1)
            plt.show()


    def voronoi_test(self, plot=True):
        node = np.array([
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
        halfedge = np.array([
            (1, 1, 1, 3, 4, 1), # 0
            (2, 1, 2, 0, 5, 1), # 1
            (3, 1, 3, 1, 6, 1), # 2
            (0, 1, 0, 2, 7, 1), # 3
            (0, 0, 7, 5, 0, 0), # 4
            (1, 0, 4, 6, 1, 0), # 5
            (2, 0, 5, 7, 2, 0), # 6
            (3, 0, 6, 4, 3, 0)],# 7
            dtype=np.int)

        domain = HalfEdgeDomain(node, halfedge, NS=1)

        bnode, idx, center, radius = domain.voronoi_mesh(n=3)
        vor = Voronoi(bnode, incremental=True)
        mesh = domain.to_halfedgemesh()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)
            mesh.find_node(axes, node=bnode, showindex=True)

            voronoi_plot_2d(vor, ax=axes)
            cs = [
                    axes.add_artist( plt.Circle(x, r, facecolor='none', edgecolor='r')) 
               for x, r in zip(center, radius)]
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)
            plt.show()


test = HalfEdgeDomainTest()
#test.advance_trimesh_test()
test.voronoi_test()

if False:
    print("halfede:")
    for i, val in enumerate(mesh.entity('halfedge')):
        print(i, ":", val)

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

    def subdomain_area_test(self):
        vertices = np.array([
            ( 0.0, 0.0), ( 1.0, 0.0), ( 1.0,  1.0), (0.0,  1.0),
            (-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0), (0.0, -1.0)], dtype=np.float)
        facets = np.array([
            (0, 1), (1, 2), (2, 3), (3, 4), 
            (4, 5), (5, 6), (6, 7), (7, 0)], dtype=np.int)
        subdomain = np.array([
            (1, 0), (1, 0), (1, 0), (1, 0),
            (1, 0), (1, 0), (1, 0), (1, 0)], dtype=np.int)
        domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
        a = domain.subdomain_area(n=1)
        print(a)

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

    def from_facets(self, plot=True):
        vertices = np.array([
            ( 0.0, 0.0), ( 1.0, 0.0), ( 1.0,  1.0), (0.0,  1.0),
            (-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0), (0.0, -1.0)], dtype=np.float)
        facets = np.array([
            (0, 1), (1, 2), (2, 3), (3, 4), 
            (4, 5), (5, 6), (6, 7), (7, 0)], dtype=np.int)
        subdomain = np.array([
            (1, 0), (1, 0), (1, 0), (1, 0),
            (1, 0), (1, 0), (1, 0), (1, 0)], dtype=np.int)

        domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
        domain.boundary_uniform_refine(n=1)
        mesh = domain.to_halfedgemesh()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)

            print("halfede:")
            for i, val in enumerate(mesh.entity('halfedge')):
                print(i, ":", val)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, color='g', showindex=True)
            plt.show()

    def voronoi_test(self, domain='square', plot=True):

        if domain == 'square':
            vertices = np.array([
                ( 0.0, 0.0), ( 1.0, 0.0), ( 1.0,  1.0), (0.0,  1.0)], dtype=np.float)
            facets = np.array([
                (0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int)
            subdomain = np.array([
                (1, 0), (1, 0), (1, 0), (1, 0)], dtype=np.int)
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            bnode, idx, center, radius = domain.voronoi_mesh(n=2)
        elif domain == 'LShape':
            vertices = np.array([
                ( 0.0, 0.0), ( 1.0, 0.0), ( 1.0,  1.0), (0.0,  1.0),
                (-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0), (0.0, -1.0)], dtype=np.float)
            fixed = np.array([1, 1, 1, 0, 1, 0, 1, 1], dtype=np.bool_)
            facets = np.array([
                (0, 1), (1, 2), (2, 3), (3, 4), 
                (4, 5), (5, 6), (6, 7), (7, 0)], dtype=np.int)
            subdomain = np.array([
                (1, 0), (1, 0), (1, 0), (1, 0),
                (1, 0), (1, 0), (1, 0), (1, 0)], dtype=np.int)

            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain,
                    fixed=fixed)
            bnode, idx, center, radius = domain.voronoi_mesh(n=2)
        elif domain == 'cirlce':
            n = 20 
            h = 2*np.pi/n
            theta = np.arange(0, 2*np.pi, h)
            vertices = np.zeros((n, 2), dtype=np.float)
            vertices[:, 0] = np.cos(theta)
            vertices[:, 1] = np.sin(theta)
            fixed = np.zeros(n, dtype=np.bool_)
            facets = np.zeros((n, 2), dtype=np.int)
            facets[:, 0] = range(0, n)
            facets[:-1, 1] = range(1, n)
            subdomain = np.zeros((n, 2), dtype=np.int)
            subdomain[:, 0] = 1
            

            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain,
                    fixed=fixed)
            bnode, idx, center, radius = domain.voronoi_mesh(n=0)

        vor = Voronoi(bnode, incremental=True)
        mesh = domain.to_halfedgemesh()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)
            mesh.find_node(axes, node=bnode, showindex=True)
            #voronoi_plot_2d(vor, ax=axes)
            cs = [
                    axes.add_artist( plt.Circle(x, r, facecolor='none', edgecolor='r')) 
               for x, r in zip(center, radius)]
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, showindex=True)
            plt.show()



test = HalfEdgeDomainTest()

if True:
    test.subdomain_area_test()

if False:
#test.advance_trimesh_test()
#test.from_facets()
    test.voronoi_test(domain='square')
#test.voronoi_test(domain='LShape')
#test.voronoi_test(domain='cirlce')

if False:
    print("halfede:")
    for i, val in enumerate(mesh.entity('halfedge')):
        print(i, ":", val)

    print('vertices:', vertices)
    print('facets:', facets)
    print('subdomain:', subdomain)
    print('fixed:', fixed)

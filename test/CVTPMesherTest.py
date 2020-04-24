#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeDomain, HalfEdgeMesh, CVTPMesher

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree

class CVTPMesherTest:

    def __init__(self):
        pass

    def uniform_boundary_meshing_test(self, plot = True):
        vertices = np.array([
            ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float)
        facets = np.array([
            (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int)
        subdomain = np.array([
            (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int)
        domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
        uniform_boundary_mesh = CVTPMesher(domain)
        uniform_boundary_mesh.uniform_boundary_meshing(n=2)
        bnode = uniform_boundary_mesh.bnode
        vor = Voronoi(bnode)
        mesh = domain.to_halfedgemesh()
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)
            mesh.find_node(axes, node=bnode, showindex=True)
            voronoi_plot_2d(vor, ax=axes)
            '''
            cs = [axes.add_artist( plt.Circle(x, r, facecolor='none',
                edgecolor='r')) for x, r in zip(center, radius)]
            '''
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, showindex=True)
            plt.show()
    def uniform_meshing_test(self, domain='square' , plot = True , interior_nodes = True,
            add_cnode = False):

        if domain == 'square':
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int)
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_boundary_meshing(n=2)
            bnode = uniform_mesh.bnode
       
        elif domain == 'LShape':
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0,  1.0),(0.0,  1.0),
                (-1.0, 1.0),(-1.0, 0.0),(-1.0, -1.0),(0.0, -1.0)],dtype=np.float)
            fixed = np.array([1, 1, 1, 0, 1, 0, 1, 1],dtype=np.bool)
            facets = np.array([
               (0, 1), (1, 2), (2, 3), (3, 4), 
               (4, 5), (5, 6), (6, 7), (7, 0)],dtype=np.int)
            subdomain = np.array([
               (1, 0), (1, 0), (1, 0), (1, 0),
               (1, 0), (1, 0), (1, 0), (1, 0)],dtype=np.int)
        
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain, fixed)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_boundary_meshing(n=2)
            bnode = uniform_mesh.bnode

        elif domain =='circle':
            n = 20
            h = 2*np.pi/n
            theta = np.arange(0, 2*np.pi, h)
            vertices = np.zeros((n, 2), dtype=np.float)
            vertices[:, 0] = np.cos(theta)
            vertices[:, 1] = np.sin(theta)
            fixed = np.zeros(n, dtype=np.bool)
            facets = np.zeros((n,2), dtype=np.int)
            facets[:, 0] = range(0, n)
            facets[:-1, 1] = range(1, n)
            subdomain = np.zeros((n, 2),dtype=np.int)
            subdomain[:, 0] = 1

            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain,
                    fixed)
            uniform_mesh= CVTPMesher(domain)
            uniform_mesh.uniform_boundary_meshing(n=0)
            bnode = uniform_mesh.bnode
        elif domain == 'partition1':
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0),(0.5,0.5)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 3),(3, 0),
                (0, 4),(4, 3),(4, 1),(4, 2)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(2, 0),(3, 0),(4, 0),
                (4, 1),(4, 3),(2, 1),(3, 2)], dtype=np.int)
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_boundary_meshing(n=3)
            bnode = uniform_mesh.bnode

        elif domain == 'partition2':
            vertices = np.array([
                ( 0.0, 0.0),(0.5, 0.0),( 1.0, 0.0),(1.0, 0.5),
                ( 1.0, 1.0),(0.5, 1.0),( 0.0, 1.0),(0.0, 0.5),
                ( 0.5, 0.5)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 3),(3, 4),
                (4, 5),(5, 6),(6, 7),(7, 0),
                (1, 8),(8, 7),(8, 3),(8, 5)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(2, 0),(2, 0),(3, 0),
                (3, 0),(4, 0),(4, 0),(1, 0),
                (1, 2),(1, 4),(3, 2),(4, 3)], dtype=np.int)
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_boundary_meshing(n=2)
            bnode = uniform_mesh.bnode
        elif domain == 'hole1':
            
            vertices = np.array([
                ( 0.0, 0.0),( 0.5, 0.0),( 1.0, 0.0),( 1.0, 0.5),
                ( 1.0, 1.0),( 0.5, 1.0),( 0.0, 1.0),( 0.0, 0.5),
                ( 0.4, 0.4),( 0.7, 0.4),( 0.7, 0.7),( 0.4, 0.7)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),( 2, 3),( 3, 4),
                (4, 5),(5, 6),( 6, 7),( 7, 0),
                (8, 9),(9,10),(10,11),(11, 8)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1,-1),(1,-1),(1,-1),(1,-1)], dtype=np.int)
            """ 
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0),
                ( 0.4, 0.4),( 0.7, 0.4),( 0.7, 0.7),( 0.4, 0.7)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),( 2, 3),( 3, 0),
                (4, 5),(5, 6),( 6, 7),( 7, 4)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1,-1),(1,-1),(1,-1),(1,-1)], dtype=np.int)
            """

            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_boundary_meshing(n=1)
            bnode = uniform_mesh.bnode
        elif domain == 'hole2':
            """
            vertices = np.array([
                ( 0.0, 0.0),( 0.5, 1.0),( 1.0, 0.0),( 1.5, 0.0),
                ( 2.0, 0.0),( 2.0, 0.5),( 2.0, 1.0),( 2.0, 1.5),
                ( 2.0, 2.0),( 1.5, 2.0),( 1.0, 2.0),( 0.5, 2.0),
                ( 0.0, 2.0),( 0.0, 1.5),( 0.0, 1.0),( 0.0, 0.5),
                ( 0.4, 0.4),( 0.4, 0.7),( 0.7, 0.7),( 0.7, 0.4),
                ( 1.2, 1.2),( 1.2, 1.5),( 1.5, 1.5),( 1.5, 1.2)],dtype=np.float)
            facets = np.array([
                ( 0, 1),( 1, 2),( 2, 3),( 3, 4),
                ( 4, 5),( 5, 6),( 6, 7),( 7, 8),
                ( 8, 9),( 9,10),(10,11),(11,12),
                (12,13),(13,14),(14,15),(15, 0),
                (16,17),(17,18),(18,19),(19,16),
                (20,21),(21,22),(22,23),(23,20)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1,-1),(1,-1),(1,-1),(1,-1),
                (1,-2),(1,-2),(1,-2),(1,-2)], dtype=np.int)
            """
            #""" 
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 2.0, 0.0),( 2.0, 1.0),
                ( 2.0, 2.0),( 1.0, 2.0),( 0.0, 2.0),( 0.0, 1.0),
                ( 0.4, 0.4),( 0.4, 0.7),( 0.7, 0.7),( 0.7, 0.4),
                ( 1.2, 1.2),( 1.2, 1.5),( 1.5, 1.5),( 1.5, 1.2)],dtype=np.float)
            facets = np.array([
                ( 0, 1),( 1, 2),( 2, 3),( 3, 4),
                ( 4, 5),( 5, 6),( 6, 7),( 7, 0),
                ( 8, 9),( 9,10),(10,11),(11, 8),
                (12,13),(13,14),(14,15),(15,12),], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1,-1),(1,-1),(1,-1),(1,-1),
                (1,-2),(1,-2),(1,-2),(1,-2)], dtype=np.int)

            #"""

            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_boundary_meshing(n=1)
            bnode = uniform_mesh.bnode

        if add_cnode == True:
            cnode = uniform_mesh.cnode
            bnode = np.append(bnode, cnode,axis=0)

        vor = Voronoi(bnode, incremental = True)
        if interior_nodes:
            uniform_mesh.uniform_init_interior_nodes()
            newnode = uniform_mesh.inode
            for k in newnode:
                vor.add_points(newnode[k])
        mesh = domain.to_halfedgemesh()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)
            mesh.find_node(axes, node=bnode, showindex=True)
            voronoi_plot_2d(vor, ax=axes)
            plt.show()
    
    def Lloyd_test(self,domain = 'square', plot = True):
        if domain == 'square':
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int)
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_meshing(refine=2)
       
        elif domain == 'LShape':
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0,  1.0),(0.0,  1.0),
               (-1.0, 1.0),(-1.0, 0.0),(-1.0, -1.0),(0.0, -1.0)],dtype=np.float)
            fixed = np.array([1, 1, 1, 0, 1, 0, 1, 1],dtype=np.bool)
            facets = np.array([
               (0, 1), (1, 2), (2, 3), (3, 4), 
               (4, 5), (5, 6), (6, 7), (7, 0)],dtype=np.int)
            subdomain = np.array([
               (1, 0), (1, 0), (1, 0), (1, 0),
               (1, 0), (1, 0), (1, 0), (1, 0),],dtype=np.int)
        
            domain = HalfEdgeDomain.from_facets(vertices, facets,
                    subdomain,fixed)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_meshing(refine=2)
        elif domain =='circle':
            n = 20
            h = 2*np.pi/n
            theta = np.arange(0, 2*np.pi, h)
            vertices = np.zeros((n, 2), dtype=np.float)
            vertices[:, 0] = np.cos(theta)
            vertices[:, 1] = np.sin(theta)
            fixed = np.zeros(n, dtype=np.bool)
            facets = np.zeros((n,2), dtype=np.int)
            facets[:, 0] = range(0, n)
            facets[:-1, 1] = range(1, n)
            subdomain = np.zeros((n, 2),dtype=np.int)
            subdomain[:, 0] = 1

            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain,
                    fixed)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_meshing(refine=0)
        elif domain == 'partition1':
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0),(0.5,0.5)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 3),(3, 0),
                (0, 4),(4, 3),(4, 1),(4, 2)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(2, 0),(3, 0),(4, 0),
                (4, 1),(4, 3),(2, 1),(3, 2)], dtype=np.int)
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_meshing(refine=3)

        elif domain == 'partition2':
            vertices = np.array([
                ( 0.0, 0.0),(0.5, 0.0),( 1.0, 0.0),(1.0, 0.5),
                ( 1.0, 1.0),(0.5, 1.0),( 0.0, 1.0),(0.0, 0.5),
                ( 0.5, 0.5)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 3),(3, 4),
                (4, 5),(5, 6),(6, 7),(7, 0),
                (1, 8),(8, 7),(8, 3),(8, 5)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(2, 0),(2, 0),(3, 0),
                (3, 0),(4, 0),(4, 0),(1, 0),
                (1, 2),(1, 4),(3, 2),(4, 3)], dtype=np.int)
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_meshing(refine=2)
        elif domain == 'hole1':
            
            vertices = np.array([
                ( 0.0, 0.0),( 0.5, 0.0),( 1.0, 0.0),( 1.0, 0.5),
                ( 1.0, 1.0),( 0.5, 1.0),( 0.0, 1.0),( 0.0, 0.5),
                ( 0.4, 0.4),( 0.7, 0.4),( 0.7, 0.7),( 0.4, 0.7)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),( 2, 3),( 3, 4),
                (4, 5),(5, 6),( 6, 7),( 7, 0),
                (8, 9),(9,10),(10,11),(11, 8)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1,-1),(1,-1),(1,-1),(1,-1)], dtype=np.int)
            """ 
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0),
                ( 0.4, 0.4),( 0.7, 0.4),( 0.7, 0.7),( 0.4, 0.7)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),( 2, 3),( 3, 0),
                (4, 5),(5, 6),( 6, 7),( 7, 4)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0),
                (1,-1),(1,-1),(1,-1),(1,-1)], dtype=np.int)
            """

            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(domain)
            uniform_mesh.uniform_meshing(refine=1)

        vor, start = uniform_mesh.voronoi()
        mesh = domain.to_halfedgemesh()
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)
            mesh.find_node(axes, node=vor.points, showindex=True)
            voronoi_plot_2d(vor, ax=axes)
            plt.show()
        
        i =0
        while i<100:
            vor = uniform_mesh.Lloyd(vor,start)
            i+=1
        
        mesh = domain.to_halfedgemesh()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)
            mesh.find_node(axes, node=vor.points, showindex=True)
            voronoi_plot_2d(vor, ax=axes)
            plt.show()
 







test = CVTPMesherTest()
#test.uniform_boundary_meshing_test()
#test.uniform_meshing_test(domain='square')
#test.uniform_meshing_test(domain='LShape')
#test.uniform_meshing_test(domain='circle')
#test.uniform_meshing_test(domain = 'partition1')
#test.uniform_meshing_test(domain = 'partition2')
#test.uniform_meshing_test(domain = 'hole1')
test.uniform_meshing_test(domain='hole2',interior_nodes=False)
#test.Lloyd_test(domain='square')
#test.Lloyd_test(domain = 'LShape')
#test.Lloyd_test(domain = 'circle')
#test.Lloyd_test(domain='partition1')
#test.Lloyd_test(domain='partition2')
#test.Lloyd_test(domain='hole1')



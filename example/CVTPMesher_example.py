#!/usr/bin/env python3
# 
import sys

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeMesh2d, CVTPMesher

from scipy.spatial import voronoi_plot_2d

class CVTPMesherExample:
    def __init__(self):
        pass
    
    def CVTMesh(self,domain = 'square', n = 2, plot = True):
        if domain == 'square':
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int)
            mesh = HalfEdgeMesh2d.from_edges(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(mesh)
            uniform_mesh.uniform_meshing(n=n)
       
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
        
            mesh = HalfEdgeMesh2d.from_edges(vertices, facets,subdomain)
            uniform_mesh = CVTPMesher(mesh,fixed)
            uniform_mesh.uniform_meshing(n=n)

        elif domain =='circle':
            N = 20
            h = 2*np.pi/N
            theta = np.arange(0, 2*np.pi, h)
            vertices = np.zeros((N, 2), dtype=np.float)
            vertices[:, 0] = np.cos(theta)
            vertices[:, 1] = np.sin(theta)
            fixed = np.zeros(N)
            facets = np.zeros((N,2), dtype=np.int)
            facets[:, 0] = range(0, N)
            facets[:-1, 1] = range(1, N)
            subdomain = np.zeros((N, 2),dtype=np.int)
            subdomain[:, 0] = 1

            mesh = HalfEdgeMesh2d.from_edges(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(mesh,fixed)
            uniform_mesh.uniform_meshing(n=n)

        elif domain == 'triangle':
            vertices = np.array([
                ( 0.0, 0.0),( 3.0, 0.0),( 2.0, 2.0)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 0)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0)], dtype=np.int)
            mesh = HalfEdgeMesh2d.from_edges(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(mesh)
            uniform_mesh.uniform_meshing(n=n)
 
        elif domain == 'trapezoid':
            vertices = np.array([
                ( 0.0, 0.0),( 4.0, 0.0),( 3.5, 4.0),(0.5,4.0)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 3),(3,0)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1,0)], dtype=np.int)
            mesh = HalfEdgeMesh2d.from_edges(vertices, facets, subdomain)
            uniform_mesh = CVTPMesher(mesh)
            uniform_mesh.uniform_meshing(n=n)
   
        mesh.print()
        vor, start = uniform_mesh.voronoi()
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)
            mesh.find_node(axes, node=vor.points, showindex=True)
            voronoi_plot_2d(vor, ax=axes)
            plt.show()
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, showindex=True)
            plt.show()

        i =0
        while i<100:
            vor = uniform_mesh.Lloyd(vor,start)
            i+=1

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=False)
            mesh.find_node(axes, node=vor.points, showindex=False)
            voronoi_plot_2d(vor, ax=axes,show_vertices = False)
            plt.show()

test = CVTPMesherExample()

n = int(sys.argv[2])
if sys.argv[1] =="S":
    test.CVTMesh(domain='square',n=n)
elif sys.argv[1] =="LS":
    test.CVTMesh(domain='LShape',n=n)
elif sys.argv[1] =="TL":
    test.CVTMesh(domain='triangle',n=n)
elif sys.argv[1] =="C":
    test.CVTMesh(domain='circle',n=n)
elif sys.argv[1] == "TP":
    test.CVTMesh(domain = 'trapezoid',n=n)


#test.CVTMesh(domain='square')
#test.CVTMesh(domain = 'LShape')
#test.CVTMesh(domain = 'circle',n=0)
#test.CVTMesh(domain = 'triangle')
#test.CVTMesh(domain = 'trapezoid')


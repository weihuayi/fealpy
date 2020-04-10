import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeDomain, HalfEdgeMesh, CVTPMesher

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree

class CVTPMesherTest:

    def __init__(self):
        pass

    def boundary_meshing_test(self, plot = True):
        vertices = np.array([
            ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float)
        facets = np.array([
            (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int)
        subdomain = np.array([
            (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int)
        domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
        bnode, hedge2bnode = CVTPMesher(domain).boundary_meshing(n=2)
        vor = Voronoi(bnode, incremental=True)
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
    def mesh_test(self, domain='square' , plot = True , interior_nodes = True):

        if domain == 'square':
            vertices = np.array([
                ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float)
            facets = np.array([
                (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int)
            subdomain = np.array([
                (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int)
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            bnode, hedge2bnode = CVTPMesher(domain).boundary_meshing(n=2)
       
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
        
            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            bnode, hedge2bnode = CVTPMesher(domain).boundary_meshing(n=2)
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

            domain = HalfEdgeDomain.from_facets(vertices, facets, subdomain)
            bnode, hedge2bnode = CVTPMesher(domain).boundary_meshing(n=0)

        vor = Voronoi(bnode, incremental = True)
        if interior_nodes:
            newnode = CVTPMesher(domain).init_interior_nodes(bnode, hedge2bnode)
            vor.add_points(newnode)
        mesh = domain.to_halfedgemesh()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)
            mesh.find_node(axes, node=bnode, showindex=True)
            voronoi_plot_2d(vor, ax=axes)
            plt.show()





test = CVTPMesherTest()
#test.boundary_meshing_test()
#test.mesh_test(domain='square')
test.mesh_test(domain='LShape')
#test.mesh_test(domain='circle')


    


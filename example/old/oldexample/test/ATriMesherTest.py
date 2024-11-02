#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import HalfEdgeDomain
from fealpy.mesh import ATriMesher

class ATriMesherTest:
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
        mesher = ATriMesher(domain)
        mesher.uniform_boundary_meshing(refine=4)
        anode = mesher.advance()
        mesh = domain.to_halfedgemesh()
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, color='k', showindex=True)
            mesh.find_node(axes, node=anode, color='r', showindex=True)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, showindex=True)
            plt.show()


test = ATriMesherTest()

if True:
    test.uniform_boundary_meshing_test(plot=True)

#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import QuadrangleMesh


class QuadrangleMeshTest:

    def __init__(self):
        pass

    def refine_RB_test(self, plot=True):

        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([[0, 1, 2, 3]], dtype=np.int)

        mesh = QuadrangleMesh(node, cell)
        mesh.uniform_refine()

        markedCell = np.array([3], dtype=np.int)

        mesh.refine_RB(markedCell)

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            mesh.print()
            plt.show()

    def Lshapetest(self, plot=True):
        node = np.array([
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (-1.0,0.0)], dtype=np.float)
        cell = np.array([[0, 1, 4, 7],[1,2,3,4],[7,4,5,6]], dtype=np.int)

        mesh = QuadrangleMesh(node, cell)
        mesh.uniform_refine(n=2)
        #mesh.node = mesh.node[np.lexsort(mesh.node[:,::-1].T)]
        '''
        l = len(mesh.node)
        node1 = np.zeros((l,2))
        node1[:9,:] = mesh.node[mesh.node[:,1] = -1]
        node1[9:18,:] = mesh.node[mesh.node[:,1] = -1]
        '''
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.node = mesh.node[np.lexsort(mesh.node[:,::-1].T)]
            mesh.find_node(axes,showindex = True)
            #mesh.find_edge(axes)
            #mesh.print()
            plt.show()


test = QuadrangleMeshTest()
#test.refine_RB_test()
test.Lshapetest()


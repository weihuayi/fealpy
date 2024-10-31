import sys
import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh, PolygonMesh, QuadrangleMesh

class TriangleMeshTest():
    def __int__():
        pass

    def lineTest(self):
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n=5)
        point = np.array([[0.2, 0.5], [0.5, 0.8], [1.6, 0.5], [1.2, 0.2]])
        segment = np.array([[0, 2], [1, 3]])

        fig = plt.figure()
        axes = fig.gca()
        for i in range(len(segment)):
            a = segment[i, 0]
            b = segment[i, 1]
            axes.plot(point[[a, b], 0], point[[a, b], 1], 'r')
        mesh.add_plot(axes)
        #mesh.find_cell(axes)
        plt.show()

    def uniform_refine_test(self):
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        mesh = TriangleMesh(node, cell)
        N,C = mesh.uniform_refine(n=2,returnim = True)

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_cell(axes)
        plt.show()

        print(N)
        print(C.toarray())

test = TriangleMeshTest()
#test.lineTest()
test.uniform_refine_test()

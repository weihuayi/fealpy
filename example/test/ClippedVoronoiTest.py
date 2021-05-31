#!/usr/bin/env python3
#
import sys 
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree

from fealpy.mesh import IntervalMesh


class ClippedVoronoiTest():
    def __init__(self):
        pass

    def voronoi_test(plot=True):

        node = np.array([
            (0.0, 0.0), (1.0, 0.0), 
            (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
        cell = np.array([
            (0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int)

        mesh = IntervalMesh(node, cell)
        mesh.uniform_refine(n=3)
        h = mesh.entity_measure('cell')

        NN = mesh.number_of_nodes()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        nh = np.zeros(NN, dtype=np.float)
        np.add.at(nh, cell[:, 0], h)
        np.add.at(nh, cell[:, 1], h)
        nh /= 2
        nh *= 3/4

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            cs = [axes.add_artist(plt.Circle(x, r, facecolor='none', edgecolor='r')) for x, r in zip(node, nh)]
            plt.show()

    def he_voronoi_test(plot=True):
        pass

test = ClippedVoronoiTest()
test.voronoi_test()

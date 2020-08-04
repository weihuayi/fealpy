#!/usr/bin/env python3
# 

import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh import HighOrderTriangleMesh, MeshFactory



class HighOrderTriangleMeshTest():

    def __init__(self):
        pass

    def show_mesh(self, p=2, plot=True):

        mf = MeshFactory()

        mesh = mf.boxmesh2d([0, 1, 0, 1], nx =2, ny=2, meshtype='tri')
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        homesh = HighOrderTriangleMesh(node, cell, p=p)
        node = homesh.entity('node')
        homesh.print()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, node=node, showindex=True, fontsize=28)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()



test = HighOrderTriangleMeshTest()

if sys.argv[1] == 'show_mesh':
    p = int(sys.argv[2])
    test.show_mesh(p=p)

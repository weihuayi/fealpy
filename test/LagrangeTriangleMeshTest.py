#!/usr/bin/env python3
# 

import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh import LagrangeTriangleMesh, MeshFactory


class LagrangeTriangleMeshTest():

    def __init__(self):
        pass

    def show_mesh(self, p=2, plot=True):

        mf = MeshFactory()

        mesh = mf.boxmesh2d([0, 1, 0, 1], nx =2, ny=2, meshtype='tri')
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        ltmesh = LagrangeTriangleMesh(node, cell, p=p)
        node = ltmesh.entity('node')
        ltmesh.print()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, node=node, showindex=True, fontsize=28)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def save_mesh(self, p=2, fname='test.vtu'):
        mf = MeshFactory()

        mesh = mf.boxmesh2d([0, 1, 0, 1], nx =2, ny=2, meshtype='tri')
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        mesh = LagrangeTriangleMesh(node, cell, p=p)
        mesh.to_vtk(fname=fname)

    def surface_mesh(self, p=2, fname='surface.vtu'):
        from fealpy.geometry import SphereSurface

        surface = SphereSurface()
        mesh = surface.init_mesh()

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        lmesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
        NC = lmesh.number_of_cells()
        a = lmesh.cell_area()
        print("NC:", NC)
        print("area:", a)
        print(sum(a))
        lmesh.to_vtk(fname=fname)


test = LagrangeTriangleMeshTest()

if sys.argv[1] == 'show_mesh':
    p = int(sys.argv[2])
    test.show_mesh(p=p)
elif sys.argv[1] == 'save_mesh':
    p = int(sys.argv[2])
    fname = sys.argv[3]
    test.save_mesh(p=p, fname=fname)
elif sys.argv[1] == 'surface_mesh':
    p = int(sys.argv[2])
    fname = sys.argv[3]
    test.surface_mesh(p=p, fname=fname)

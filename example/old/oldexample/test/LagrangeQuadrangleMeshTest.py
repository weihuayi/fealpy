#!/usr/bin/env python3
# 

import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh import LagrangeQuadrangleMesh, MeshFactory


class LagrangeQuadrangleMeshTest():

    def __init__(self):
        pass

    def show_mesh(self, p=2, plot=True):

        mf = MeshFactory()

        mesh = mf.boxmesh2d([0, 1, 0, 1], nx =2, ny=2, meshtype='quad')
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        lqmesh = LagrangeQuadrangleMesh(node, cell[:, [0, 3, 1, 2]], p=p)
        NN = lqmesh.number_of_nodes()
        node = lqmesh.entity('node')
        lqmesh.print()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.ds.edge = lqmesh.ds.edge
            mesh.ds.edge2cell = lqmesh.ds.edge2cell
            mesh.add_plot(axes)
            mesh.find_node(axes, node=node, showindex=True, fontsize=28)
            #mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def save_mesh(self, p=2, fname='test.vtu'):
        mf = MeshFactory()

        mesh = mf.boxmesh2d([0, 1, 0, 1], nx=2, ny=2, meshtype='quad')
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        mesh = LagrangeQuadrangleMesh(node, cell[:, [0, 3, 1, 2]], p=p)
        mesh.to_vtk(fname=fname)

    def surface_mesh(self, p=2, fname='surface.vtu'):
        from fealpy.geometry import SphereSurface

        surface = SphereSurface()
        node = np.array([
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1)], dtype=np.float64)
        cell = np.array([
            (0, 1, 4, 5),
            (6, 7, 2, 3),
            (2, 3, 0, 1),
            (4, 5, 6, 7),
            (1, 3, 5, 7),
            (2, 0, 6, 4)], dtype=np.int_)

        lmesh = LagrangeQuadrangleMesh(node, cell, p=p, surface=surface)

        lmesh.uniform_refine(n=1)
        lmesh.to_vtk(fname=fname)

    def surface_area(self, p=2):
        from fealpy.geometry import SphereSurface

        surface = SphereSurface()
        mesh = surface.init_mesh(meshtype='quad', p=p)
        e = 0
        maxit = 4
        a_e = (4*np.pi)
        for i in range(maxit):
            print(i, ":")
            NC = mesh.number_of_cells()
            a = mesh.cell_area()
            a = sum(a)
            e_new = abs(a-a_e)

            order = np.log2(e/e_new)
            e = e_new
            print("e:", e)
            print("order:", order)
            if i < maxit - 1:
                mesh.uniform_refine()

        mesh.to_vtk(fname='quad.vtu')
        print(mesh.ds.V)


test = LagrangeQuadrangleMeshTest()

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
elif sys.argv[1] == 'surface_area':
    p = int(sys.argv[2])
    test.surface_area(p=p)

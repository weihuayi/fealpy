#!/usr/bin/env python3
# 
import sys
import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from fealpy.writer import MeshWriter
from fealpy.mesh import HalfEdgeMesh2d
from fealpy.mesh import HalfEdgeMesh
from fealpy.mesh import TriangleMesh, PolygonMesh, QuadrangleMesh


class HalfEdgeMesh2dTest:
    def __init__(self):
        pass

    def data_structure(self, plot=True):
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.print()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def from_edges(self, plot=True):

        node = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float)
        edge = np.array([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int)
        edge2subdomain = np.array([(1, 0), (1, 0), (1, 0), (1, 0)],
                dtype=np.int)

        mesh = HalfEdgeMesh2d.from_edges(node, edge, edge2subdomain)
        mesh.print()
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def tiling_test(self, plot=True):
        pass

    def dynamic_array(self):
        pass

    def cell_to_node(self):
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        print('node:', node)
        print('cell:', cell)
        print('cellLocation:', cellLocation)
        mesh = PolygonMesh(node, cell, cellLocation)

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)

        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.print()
        cell, cellLocation = mesh.entity('cell')
        print('cell:', cell)
        print('cellLocation:', cellLocation)
        plt.show()

    def refine_halfedge(self, plot=True):
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)

        isMarkedCell = np.array([0, 1, 0, 0, 0], dtype=np.bool_)
        isMarkedHEdge = mesh.mark_halfedge(isMarkedCell)
        mesh.refine_halfedge(isMarkedHEdge)
        mesh.print()
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, showindex=True)
            #mesh.find_cell(axes, showindex=True)
            plt.show()

    def refine_poly(self, plot=True):
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.uniform_refine(n=2)
        print(mesh.number_of_nodes())

        #fig = plt.figure()
        #axes = fig.gca()
        #mesh.add_plot(axes)
        #mesh.find_node(axes, showindex=True)
        #mesh.find_cell(axes, showindex=True)

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        if True:
            isMarkedCell = mesh.mark_helper([2])
            mesh.refine_poly(isMarkedCell)

        if False:
            isMarkedCell = mesh.mark_helper([6])
            mesh.refine_poly(isMarkedCell, dflag=False)

        if False:
            isMarkedCell = mesh.mark_helper([3])
            mesh.refine_poly(isMarkedCell, dflag=False)

        if False:
            isMarkedCell = mesh.mark_helper([1, 5])
            mesh.refine_poly(isMarkedCell)

        if False:
            isMarkedCell = mesh.mark_helper([1, 12])
            mesh.refine_poly(isMarkedCell, dflag=False)

        if False:
            isMarkedCell = mesh.mark_helper([0, 21])
            mesh.refine_poly(isMarkedCell, dflag=False)

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

            if 0:
                NAC = mesh.number_of_all_cells() # 包括外部区域和洞
                cindex = range(mesh.ds.cellstart, NAC)
                fig = plt.figure()
                axes = fig.gca()
                #mesh.add_plot(axes)
                mesh.add_halfedge_plot(axes, showindex=True)
                mesh.find_node(axes, showindex=True)
                mesh.find_cell(axes, showindex=True, multiindex=cindex)

                NN = mesh.number_of_nodes()
                nindex = np.zeros(NN, dtype=np.int)
                halfedge = mesh.ds.halfedge
                nindex[halfedge[:, 0]] = mesh.get_data('halfedge', 'level')
                cindex = mesh.get_data('cell', 'level')
                fig = plt.figure()
                axes = fig.gca()
                #mesh.add_plot(axes)
                mesh.find_node(axes, showindex=True, multiindex=nindex)
                #mesh.find_cell(axes, showindex=True, multiindex=cindex)
                plt.show()
        else:
            return mesh

    def adaptive_poly(self, plot=True):

        """"
        initial mesh
        """
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)

        #fig = plt.figure()
        #axes = fig.gca()
        #mesh.add_plot(axes)
        #mesh.find_node(axes, showindex=True)
        #mesh.find_cell(axes, showindex=True)

        NE = mesh.number_of_edges()
        nC = mesh.number_of_cells()

        """
        refined mesh
        """
        aopts = mesh.adaptive_options(method='numrefine',maxcoarsen=3,HB=True)
        eta = [2,2,2,2,2]

        mesh.adaptive(eta, aopts)
        print('r',aopts['HB'])

        #fig = plt.figure()
        #axes = fig.gca()
        #mesh.add_plot(axes)
        #mesh.find_node(axes, showindex=True)
        #mesh.find_cell(axes, showindex=True)
        #plt.show()

#        mesh.from_mesh(mesh)
        """
        coarsened mesh
        """
#        eta = [0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1]
#        #eta = [0,0,0,0,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2]
#
#        mesh.adaptive(eta, aopts)
#        print('c',aopts['HB'])
#        print(mesh.ds.hcell)
        #fig = plt.figure()
        #axes = fig.gca()
        #mesh.add_plot(axes)
        #mesh.find_node(axes, showindex=True)
        #mesh.find_cell(axes, showindex=True)
        #plt.show()

        if plot:

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, showindex=True)
            #mesh.find_cell(axes, showindex=True)
            plt.show()
        if 0:
            NAC = mesh.number_of_all_cells() # 包括外部区域和洞
            cindex = range(mesh.ds.cellstart, NAC)
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True, multiindex=cindex)

            NN = mesh.number_of_nodes()
            nindex = np.zeros(NN, dtype=np.int)
            halfedge = mesh.ds.halfedge
            nindex[halfedge[:, 0]] = mesh.get_data('halfedge', 'level')
            cindex = mesh.get_data('cell', 'level')
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True, multiindex=nindex)
            mesh.find_cell(axes, showindex=True, multiindex=cindex)
            plt.show()
        else:
            return mesh

    def tri_cut_graph(self, fname, weight = None):
        data = sio.loadmat(fname)
        node = np.array(data['node'], dtype=np.float64)
        cell = np.array(data['elem'] - 1, dtype=np.int_)

        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.ds.NV = 3

        gamma = mesh.tri_cut_graph(weight = weight)

        writer = MeshWriter(mesh)
        writer.write(fname='test.vtu')
        for i, index in enumerate(gamma):
            writer = MeshWriter(mesh, etype='edge', index=index)
            writer.write(fname='test'+str(i)+'.vtu')

    def quad_refine(self, plot=True):
        cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        mesh = QuadrangleMesh(node, cell)

        mesh0 = HalfEdgeMesh.from_mesh(mesh)

        isMarkedCell = np.array([0, 0, 1], dtype=np.bool_)

        mesh0.refine_quad(isMarkedCell)

        mesh1 = HalfEdgeMesh2d.from_mesh(mesh)
        mesh1.ds.NV = 4
        mesh1.node = mesh0.node
        halfedge = mesh0.ds.halfedge[:, :-1]
        mesh1.ds.reinit(halfedge, mesh0.ds.subdomain)

        NE = mesh1.ds.NE
        color = 3*np.ones(NE*2, dtype = np.int_)
        color[0]=1
        while (color==3).any():
            red = color == 1
            gre = color == 0
            color[halfedge[red][:, [2,3,4]]] = 0
            color[halfedge[gre][:, [2,3,4]]] = 1
            print(color)


        color[[24, 27]] = 2
        mesh1.hedgecolor = color
        isMarkedCell = np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.bool_)
        isMarkedHEdge = mesh1.mark_halfedge(isMarkedCell, method = 'quad')
        mesh0.ds.hedge = np.arange(NE*2)[isMarkedHEdge]
        print(np.where(isMarkedHEdge))
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh0.add_plot(axes)
            mesh0.add_halfedge_plot(axes, showindex=True)
            mesh0.find_cell(axes, showindex=True)
            plt.show()


test = HalfEdgeMesh2dTest()

if sys.argv[1] == "data_structure":
    test.data_structure()
elif sys.argv[1] == 'from_edges':
    test.from_edges()
elif sys.argv[1] == 'refine_halfedge':
    test.refine_halfedge()
elif sys.argv[1] == 'refine_poly':
    test.refine_poly(plot=True)
elif sys.argv[1] == 'adaptive_poly':
    mesh = test.adaptive_poly()
elif sys.argv[1] == 'cell_to_node':
    mesh = test.cell_to_node()
elif sys.argv[1] == 'read':
    fname = sys.argv[2]
    test.tri_cut_graph(fname, weight = 'length')
elif sys.argv[1] == 'quad_refine':
    test.quad_refine()




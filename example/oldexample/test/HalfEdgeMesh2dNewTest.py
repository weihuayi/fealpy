#!/usr/bin/env python3
# 
import sys
import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from fealpy.writer import MeshWriter
from fealpy.mesh import HalfEdgeMesh2d
#from fealpy.mesh import HalfEdgeMesh
from fealpy.mesh import TriangleMesh, PolygonMesh, QuadrangleMesh


class HalfEdgeMesh2dTest:
    def __init__(self):
        pass

    def interpolation(self, n=2, plot=True):
        from fealpy.pde.poisson_2d import CosCosData
        from fealpy.functionspace import ConformingVirtualElementSpace2d

        pde = CosCosData()
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)
        cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
        mesh = QuadrangleMesh(node, cell)
        #mesh = PolygonMesh.from_mesh(mesh)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.uniform_refine(n=n)

        #mesh.print()

        space = ConformingVirtualElementSpace2d(mesh, p=1)
        uI = space.interpolation(pde.solution)
        up = space.project_to_smspace(uI)
        error = space.integralalg.L2_error(pde.solution, up)
        print(error)


        if plot:
            fig = plt.figure()
            axes = fig.gca()
            #mesh.add_halfedge_plot(axes, showindex=True)
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()



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
        mesh.init_level_info()

        if False:
            NC = mesh.ds.number_of_all_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[:] = True
            mesh.refine_poly(isMarkedCell)
            NC = mesh.ds.number_of_all_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[[1,2,3,4,7,10]] = True
            i = 0
            mesh.refine_poly(isMarkedCell)
            NC = mesh.ds.number_of_all_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[[17, 19, 36, 24, 14, 21, 34]] = True
            i = 1
            mesh.refine_poly(isMarkedCell)
            NC = mesh.ds.number_of_all_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[[14, 16]] = True
            i = 1
            mesh.refine_poly(isMarkedCell)


        if True:
            c = np.array([1,1])
            r = 0.8
            h = 1e-2
            k=0
            NB = 0
            while k<8:
                halfedge = mesh.ds.halfedge
                halfedge1 = halfedge[:, 3]
                node = mesh.node
                flag = node-c
                flag = flag[:,0]**2+flag[:,1]**2
                flag = flag<=r**2
                flag1 = flag[halfedge[:, 0]].astype(int)
                flag2 = flag[halfedge[halfedge1, 0]].astype(int)
                markedge = flag1+flag2==1
                markedcell = halfedge[markedge, 1]
                markedcell = np.unique(markedcell)
                cell = np.unique(halfedge[:,1])
                nc = cell.shape[0]
                markedcell1 = np.zeros(nc)
                markedcell1[markedcell] = 1
                print('makee',markedcell)
                mesh.refine_poly(markedcell1.astype(np.bool_))
                k+=1
                print('循环',k,'次***************************')

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        #mesh.find_cell(axes, showindex=True)
        #mesh.add_halfedge_plot(axes, showindex=True)
        plt.show()

    def coarsen_poly(self, plot=True):

        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.init_level_info()

        isMarkedCell = np.array([0,0,0,1,0], dtype=np.bool_)
        mesh.refine_poly(isMarkedCell)


        NC = mesh.number_of_all_cells()
        isMarkedCell = np.zeros(NC, dtype=np.bool_)
        isMarkedCell[[1, 2, 3, 4, 5, 6]] = True
        mesh.coarsen_poly(isMarkedCell)

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        mesh.add_halfedge_plot(axes, showindex=True)
        plt.show()

    def tri_cut_graph(self, fname, weight = None):
        data = sio.loadmat(fname)
        node = np.array(data['node'], dtype=np.float64)
        cell = np.array(data['elem'] - 1, dtype=np.int_)

        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh, closed=True)
        mesh.ds.NV = 3

        gamma = mesh.tri_cut_graph(weight = weight)

        writer = MeshWriter(mesh)
        writer.write(fname='test.vtu')
        for i, index in enumerate(gamma):
            writer = MeshWriter(mesh, etype='edge', index=index)
            writer.write(fname='test'+str(i)+'.vtu')

    def refine_quad(self, l, plot=True):
        cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        mesh = QuadrangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.init_level_info()
        halfedge = mesh.ds.halfedge
        NE = mesh.ds.NE
        color = 3*np.ones(NE*2, dtype = np.int_)
        color[1]=1
        while (color==3).any():
            red = color == 1
            gre = color == 0
            color[halfedge[red][:, [2,3,4]]] = 0
            color[halfedge[gre][:, [2,3,4]]] = 1
        colorlevel = ((color==1) | (color==2)).astype(np.int_)
        mesh.hedgecolor = {'color':color, 'level':colorlevel}
        c = np.array([1,10000.5])
        r = 10000
        h = 1e-2
        k=0
        NB = 0
        fig = plt.figure()
        axes = fig.gca()
        plt.ion()
        while k<l:
            halfedge = mesh.ds.halfedge
            halfedge1 = halfedge[:, 3]
            node = mesh.node
            flag = node-c
            flag = flag[:,0]**2+flag[:,1]**2
            flag = flag<=r**2
            flag1 = flag[halfedge[:, 0]].astype(int)
            flag2 = flag[halfedge[halfedge1, 0]].astype(int)
            markedge = flag1+flag2==1
            markedcell = halfedge[markedge, 1]
            markedcell = np.unique(markedcell)
            cell = np.unique(halfedge[:,1])
            nc = cell.shape[0]
            markedcell1 = np.zeros(nc)
            markedcell1[markedcell] = 1
            print('makee',markedcell)
            mesh.refine_quad(markedcell1.astype(np.bool_))
            k+=1
            print('循环',k,'次***************************')
            print(np.c_[np.arange(mesh.ds.NE*2), mesh.hedgecolor['color'],
                mesh.hedgecolor['level']])
            if plot:
                plt.cla()
                mesh.add_plot(axes)
                plt.pause(0.001)
        plt.ioff()
        plt.show()

    def coarsen_quad(self, plot=True):
        cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        mesh = QuadrangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.init_level_info()
        halfedge = mesh.ds.halfedge
        NE = mesh.ds.NE
        color = 3*np.ones(NE*2, dtype = np.int_)
        color[1]=1
        while (color==3).any():
            red = color == 1
            gre = color == 0
            color[halfedge[red][:, [2,3,4]]] = 0
            color[halfedge[gre][:, [2,3,4]]] = 1
        colorlevel = ((color==1) | (color==2)).astype(np.int_)
        mesh.hedgecolor = {'color':color, 'level':colorlevel}

        for i in range(1):
            NC = mesh.number_of_all_cells()
            isMarkedCell = np.ones(NC, dtype=np.bool_)
            mesh.refine_quad(isMarkedCell)
        NC = mesh.number_of_all_cells()
        isMarkedCell = np.zeros(NC, dtype=np.bool_)
        isMarkedCell[[3, 5, 7, 8]] = True
        #isMarkedCell[[1, 3, 12,18, 20, 17, 16, 9, 7, 2, 4,6, 7, 8, 9 , 27, 15,
         #   8, 23, 32, 5, 24, 16, 29, 4, 7, 28, 11, 14, 22, 31, 19]] = True
        mesh.coarsen_quad(isMarkedCell)
        print(np.c_[np.arange(len(mesh.ds.halfedge)), mesh.ds.halfedge])
        if 0:
            NC = mesh.number_of_all_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[[20,21,23,22,18,24,26,11,1,7,13,3,8,15,12,17]] = True
            mesh.coarsen_quad(isMarkedCell)

        for i in range(0):
            print(i, '*************************************************')
            NC = mesh.number_of_all_cells()
            isMarkedCell = np.ones(NC, dtype=np.bool_)
            mesh.refine_quad(isMarkedCell)
        print(np.where(mesh.hedgecolor['color']==2)) 
        #print(np.c_[np.arange(mesh.ds.NE*2), mesh.hedgecolor['color'],
         #   mesh.hedgecolor['level']])
        #print(np.c_[np.arange(mesh.ds.NE*2), mesh.halfedgedata['level']])
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.add_halfedge_plot(axes, showindex=True)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def refine_tri(self, maxit = 2, method = 'rg', plot=True, rb=True):
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)

        if False:
            mesh = TriangleMesh(node, cell)
            mesh = HalfEdgeMesh.from_mesh(mesh)
            mesh.ds.cell2hedge = np.array([0, 3, 2, 11, 10])
            isMarkedCell = np.array([0, 1, 0, 0 ,1], dtype = np.bool_)
            #mesh.refine_triangle_rbg(isMarkedCell)

            mesh.ds.NV = 3
            cell = mesh.ds.cell_to_node()
            node = mesh.entity('node')

            mesh = TriangleMesh(node, cell)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.init_level_info()
            if False:
                fig = plt.figure()
                axes = fig.gca()
                mesh.add_plot(axes)
                mesh.add_halfedge_plot(axes, showindex=True)
                mesh.find_node(axes, showindex=True)
                mesh.find_cell(axes, showindex=True)
                plt.show()

            NE = mesh.ds.NE
            color = np.zeros(NE*2, dtype=np.int_)
            if method == 'rg':
                color[[4, 13, 17, 28]] = 1
                color[[23,27]] = 2
                color[[22, 26]] = 3
                mesh.hedgecolor=  color
                isMarkedCell = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0 ,0],
                        dtype=np.bool_)
                mesh.refine_triangle_rg(isMarkedCell)
            else:
                color[[2,3,10,11]] = 1
                mesh.hedgecolor = color
                isMarkedCell = np.array([0, 1, 1, 0, 0], dtype=np.bool_)
                #isMarkedCell = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ,0],
                 #       dtype=np.bool_)
                mesh.refine_triangle_nvb(isMarkedCell)
                mesh.print()
            if plot:
                fig = plt.figure()
                axes = fig.gca()
                mesh.add_plot(axes)
                mesh.add_halfedge_plot(axes, showindex=True)
                mesh.find_node(axes, showindex=True)
                mesh.find_cell(axes, showindex=True)
                plt.show()
        if True:
            mesh = TriangleMesh(node, cell)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.init_level_info()
            NE = mesh.ds.NE
            color = np.zeros(NE*2, dtype=np.int_)
            if method == 'nvb':
                color[[2,3,10,11]] = 1
            mesh.hedgecolor = color
            c = np.array([0.8,0.8])
            r = 0.9
            h = 1e-2
            k=0
            NB = 0
            start = time.time()
            while k<maxit:
                halfedge = mesh.ds.halfedge
                halfedge1 = halfedge[:, 3]
                node = mesh.node
                flag = node-c
                flag = flag[:,0]**2+flag[:,1]**2
                flag = flag<=r**2
                flag1 = flag[halfedge[:, 0]].astype(int)
                flag2 = flag[halfedge[halfedge1, 0]].astype(int)
                markedge = flag1+flag2==1
                markedcell = halfedge[markedge, 1]
                markedcell = np.unique(markedcell)
                cell = np.unique(halfedge[:,1])
                nc = cell.shape[0]
                markedcell1 = np.zeros(nc)
                markedcell1[markedcell] = 1
                if method == 'rg':
                    mesh.refine_triangle_rg(markedcell1.astype(np.bool_))
                else:
                    mesh.refine_triangle_nvb(markedcell1.astype(np.bool_))
                k+=1
                print('循环',k,'次***************************')
            end = time.time()
            print('用时', end-start)
            if plot:
                fig = plt.figure()
                axes = fig.gca()
                nindex = mesh.nodedata['level']
                mesh.add_plot(axes)
                #mesh.add_halfedge_plot(axes, showindex=True)
                #mesh.find_node(axes, showindex=True, multiindex=nindex)
                #mesh.find_cell(axes, showindex=True)
                #print(np.c_[np.arange(len(mesh.hedgecolor)), mesh.hedgecolor])
                plt.show()

    def coarsen_tri(self, maxit = 2, method = 'rg', plot=True, rb=True):
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)

        if True:
            mesh = TriangleMesh(node, cell)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.init_level_info()
            isMarkedCell = np.array([0,1,0,0,1], dtype=np.bool_)
            NE = mesh.ds.NE
            color = np.zeros(NE*2, dtype=np.int_)
            mesh.hedgecolor = color
            mesh.refine_triangle_rg(isMarkedCell)
            NE = mesh.ds.NE
            color = np.zeros(NE*2, dtype=np.int_)
            if 0:
                fig = plt.figure()
                axes = fig.gca()
                mesh.add_plot(axes)
                mesh.add_halfedge_plot(axes, showindex=True)
                mesh.find_node(axes, showindex=True)
                mesh.find_cell(axes, showindex=True)
                plt.show()
            if method == 'rg':
                if 1:
                    NC = mesh.number_of_all_cells()
                    isMarkedCell = np.zeros(NC, dtype=np.bool_)
                    isMarkedCell[[1, 2, 5, 6, 7, 9]] = True
                    print('*************lll*********')
                    mesh.coarsen_triangle_rg(isMarkedCell)
                if 0:
                    NC = mesh.number_of_all_cells()
                    isMarkedCell = np.zeros(NC, dtype=np.bool_)
                    isMarkedCell[[0, 1, 2, 3, 4, 5, 6, 7]] = True
                    print('*************lll*********')
                    mesh.coarsen_triangle_rg(isMarkedCell)
                    NC = mesh.number_of_all_cells()
                    isMarkedCell = np.ones(NC, dtype=np.bool_)
                    print('*************lll*********')
                    #mesh.refine_triangle_rg(isMarkedCell)

            else:
                print('mm')
                color[[2,3,10,11]] = 1
                mesh.hedgecolor = color
                isMarkedCell = np.array([0, 1, 1, 0, 0], dtype=np.bool_)
                #isMarkedCell = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ,0],
                #        dtype=np.bool_)
                mesh.refine_triangle_nvb(isMarkedCell)
            if 0:
                NC = mesh.number_of_all_cells()
                isMarkedCell = np.zeros(NC, dtype=np.bool_)
                isMarkedCell[[1, 3, 4, 8, 11, 12]] = True
                print('*************lll*********')
                mesh.coarsen_triangle_nvb(isMarkedCell)
            if 0:
                NC = mesh.number_of_all_cells()
                isMarkedCell = np.zeros(NC, dtype=np.bool_)
                isMarkedCell[[1, 6, 11]] = True
                mesh.coarsen_triangle_nvb(isMarkedCell)
            if 0:
                NC = mesh.number_of_all_cells()
                isMarkedCell = np.zeros(NC, dtype=np.bool_)
                isMarkedCell[[1, 2, 3]] = True
                mesh.refine_triangle_nvb(isMarkedCell)
                #mesh.print()
            if plot:
                fig = plt.figure()
                axes = fig.gca()
                mesh.add_plot(axes)
                mesh.add_halfedge_plot(axes, showindex=True)
                mesh.find_node(axes, showindex=True)
                mesh.find_cell(axes, showindex=True)
                plt.show()

    def convexity(self):
        node = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [1, 3], 
            [1, 4], [2, 4], [2, 5], [0, 5], [0, 2]], dtype
                = np.float)
        cell = np.array([0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9, 10, 4], dtype=np.int_)
        cellLocation = np.array([0, 6, 13], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.init_level_info()
        mesh.convexity()

        mesh.print()
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        mesh.add_halfedge_plot(axes, showindex=True)
        plt.show()

    def animation_plot(self, method='quad', plot=True):

        if method=='quad':
            cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int)
            node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
            mesh = QuadrangleMesh(node, cell)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.init_level_info()
            halfedge = mesh.ds.halfedge
            cstart = mesh.ds.cellstart
            NE = mesh.ds.NE
            color = 3*np.ones(NE*2, dtype = np.int_)
            color[1]=1
            while (color==3).any():
                red = color == 1
                gre = color == 0
                color[halfedge[red][:, [2,3,4]]] = 0
                color[halfedge[gre][:, [2,3,4]]] = 1
            colorlevel = ((color==1) | (color==2)).astype(np.int_)
            mesh.hedgecolor = {'color':color, 'level':colorlevel}
            mesh.ds.NV = 4
        elif method=='rg':
            cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
            node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
            mesh = TriangleMesh(node, cell)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.init_level_info()
            halfedge = mesh.ds.halfedge
            cstart = mesh.ds.cellstart
            NE = mesh.ds.NE
            color = np.zeros(NE*2, dtype=np.int_)
            mesh.hedgecolor = color
            #mesh.ds.NV = 3
        elif method=='nvb':
            cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
            node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
            mesh = TriangleMesh(node, cell)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.init_level_info()
            halfedge = mesh.ds.halfedge
            cstart = mesh.ds.cellstart
            NE = mesh.ds.NE
            print('hahahah')
            #color = np.zeros(NE*2, dtype=np.int_)
            #color[[2,3,10,11]] = 1
            #mesh.hedgecolor = color
        elif method=='poly':
            node = np.array([
                (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),], dtype=np.float)
            cell = np.array([0,2,3,0,3,1], dtype=np.int)
            cellLocation = np.array([0, 3, 6], dtype=np.int)

            mesh = PolygonMesh(node, cell, cellLocation)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.init_level_info()
            halfedge = mesh.ds.halfedge
            cstart = mesh.ds.cellstart
        r = 0.5
        h = 1e-2
        k=0
        N = 19
        fig = plt.figure()
        axes = fig.gca()
        plt.ion()
        for i in range(N):
            c = np.array([i*(2/N), 0.8])
            k=0
            sta1 = time.time()
            while True:
                halfedge = mesh.ds.halfedge
                pre = halfedge[:, 3]
                node = mesh.entity('node')
                flag = np.linalg.norm(node-c, axis=1)<r
                flag1 = flag[halfedge[:, 0]].astype(int)
                flag2 = flag[halfedge[pre, 0]].astype(int)
                isMarkedHEdge = flag1+flag2==1
                NC = mesh.number_of_all_cells()
                isMarkedCell = np.zeros(NC, dtype=np.bool_)
                isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
                isMarkedCell[cstart:] = isMarkedCell[cstart:] & (mesh.cell_area()>h**2)
                if (~isMarkedCell[cstart:]).all():
                    break
                if method=='quad':
                    mesh.refine_quad(isMarkedCell)
                elif method=='rg':
                    mesh.refine_triangle_rg(isMarkedCell)
                elif method=='nvb':
                    mesh.refine_triangle_nvb(isMarkedCell)
                elif method=='poly':
                    mesh.refine_poly(isMarkedCell)
                k+=1
                print('加密',k,i,'次***************************')
            k=0
            sta2 = time.time()

            aa = 2*i
            bb = 2*i+1
            plt.cla()
            mesh.add_plot(axes, linewidths = 0.4)
            #fig.savefig('%f.png' %aa, dpi=600, bbox_inches='tight')
            plt.pause(0.01)

            while k<10:
                halfedge = mesh.ds.halfedge
                pre = halfedge[:, 3]
                node = mesh.entity('node')
                flag = np.linalg.norm(node-c, axis=1)<r
                flag1 = flag[halfedge[:, 0]].astype(int)
                flag2 = flag[halfedge[pre, 0]].astype(int)
                isMarkedHEdge = flag1+flag2==1
                NC = mesh.number_of_all_cells()
                isMarkedCell = np.zeros(NC, dtype=np.bool_)
                isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
                isMarkedCell[cstart:] = ~isMarkedCell[cstart:] & (mesh.cell_area()<0.5)
                if method=='quad':
                    mesh.coarsen_quad(isMarkedCell)
                elif method=='rg':
                    mesh.coarsen_triangle_rg(isMarkedCell)
                elif method=='nvb':
                    mesh.coarsen_triangle_nvb(isMarkedCell)
                elif method=='poly':
                    mesh.coarsen_poly(isMarkedCell)
                if (~isMarkedCell).all():
                    break
                k+=1
                print('循环',k,'次***************************')
            plt.cla()
            mesh.add_plot(axes, linewidths = 0.4)
            #fig.savefig('%f.png' %bb, dpi=600, bbox_inches='tight')
            sta3 = time.time()
            plt.pause(0.01)
            print('加密时间:', sta2-sta1)
            print('粗化时间:', sta3-sta2)
        plt.ioff()
        plt.show()

    def adaptive(self, method='poly', plot=True):

        """"
        initial mesh
        """
        Mesh = initMesh(method=method)
        mesh = Mesh.mesh

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)

        """
        refined mesh
        """
        aopts = mesh.adaptive_options(method='numrefine',maxcoarsen=3,HB=True)

        eta = [0,2, 0, 0 , 3]
        mesh.adaptive(eta, aopts, method=method)
        #eta = np.zeros(mesh.number_of_cells(), dtype=np.int_)
        #eta[[0, 2]] = 1
        #eta[16] = 2
        #mesh.adaptive(eta, aopts, method=method)
        #eta[[5, 11, 6, 20, 15, 12]] = -1

        #mesh.adaptive(eta, aopts, method=method)
        print('r',aopts['HB'])
        print('r',np.c_[np.arange(len(aopts['numrefine'])),
            aopts['numrefine']])
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        plt.show()

        """
        coarsened mesh
        """
        #eta = [0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1]

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        plt.show()

class initMesh():
    def __init__(self, method='rg'):
        if (method == 'rg') | (method == 'nvb'):
            self.mesh = self.triMesh(method=method)
        elif method == 'quad':
            self.mesh = self.quadMesh()
        elif method == 'poly':
            self.mesh = self.polyMesh()

    def triMesh(self, method='rg'):
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.init_level_info()
        if method == 'rg':
            halfedge = mesh.ds.halfedge
            cstart = mesh.ds.cellstart
            NE = mesh.ds.NE
            color = np.zeros(NE*2, dtype=np.int_)
            mesh.hedgecolor = color
        elif method=='nvb':
            halfedge = mesh.ds.halfedge
            cstart = mesh.ds.cellstart
            NE = mesh.ds.NE
            color = np.zeros(NE*2, dtype=np.int_)
            color[[2,3,10,11]] = 1
            mesh.hedgecolor = color
        return mesh

    def quadMesh(self):
        cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        mesh = QuadrangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.init_level_info()

        halfedge = mesh.ds.halfedge
        cstart = mesh.ds.cellstart
        NE = mesh.ds.NE
        color = 3*np.ones(NE*2, dtype = np.int_)
        color[1]=1
        while (color==3).any():
            red = color == 1
            gre = color == 0
            color[halfedge[red][:, [2,3,4]]] = 0
            color[halfedge[gre][:, [2,3,4]]] = 1
        colorlevel = ((color==1) | (color==2)).astype(np.int_)
        mesh.hedgecolor = {'color':color, 'level':colorlevel}
        return mesh

    def polyMesh(self):
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.init_level_info()


        return mesh


test = HalfEdgeMesh2dTest()
if sys.argv[1] == "data_structure":
    test.data_structure()
elif sys.argv[1] == 'from_edges':
    test.from_edges()
elif sys.argv[1] == 'refine_halfedge':
    test.refine_halfedge()
elif sys.argv[1] == 'refine_poly':
    test.refine_poly(plot=True)
elif sys.argv[1] == 'adaptive':
    mesh = test.adaptive(method=sys.argv[2])
elif sys.argv[1] == 'cell_to_node':
    mesh = test.cell_to_node()
elif sys.argv[1] == 'read':
    fname = sys.argv[2]
    test.tri_cut_graph(fname, weight = 'length')
elif sys.argv[1] == 'refine_quad':
    test.refine_quad(int(sys.argv[2]))
elif sys.argv[1] == 'refine_tri':
    test.refine_tri(maxit = int(sys.argv[3]), method = sys.argv[2])
elif sys.argv[1] == "interpolation":
    n = int(sys.argv[2])
    test.interpolation(n=n, plot=False)
elif sys.argv[1] == "coarsen_poly":
    test.coarsen_poly()
elif sys.argv[1] == "convexity":
    test.convexity()
elif sys.argv[1] == "coarsen_quad":
    test.coarsen_quad()
elif sys.argv[1] == "coarsen_tri":
    test.coarsen_tri(maxit = int(sys.argv[3]), method = sys.argv[2])
elif sys.argv[1] == 'animation_plot':
    test.animation_plot(method=sys.argv[2])

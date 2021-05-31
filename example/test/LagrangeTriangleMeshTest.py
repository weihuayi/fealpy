#!/usr/bin/env python3
# 

import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh import LagrangeTriangleMesh,TriangleMesh, MeshFactory


class LagrangeTriangleMeshTest():

    def __init__(self):
        pass

    def show_mesh(self, p=2, plot=True):

        mf = MeshFactory()

        mesh = mf.boxmesh2d([0, 1, 0, 1], nx =1, ny=1, meshtype='tri')
        node = mesh.entity('node')
        #cell = mesh.entity('cell')
        cell = np.array([[0,2,1]])

        ltmesh = LagrangeTriangleMesh(node, cell, p=p)
        NN = ltmesh.number_of_nodes()

        mesh.ds.edge = ltmesh.ds.edge
        mesh.ds.edge2cell = ltmesh.ds.edge2cell

        node = ltmesh.entity('node')
        #ltmesh.print()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, node=node, showindex=True, fontsize=28)
            #mesh.find_edge(axes, showindex=True)
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
        from fealpy.geometry import SphereSurface, EllipsoidSurface, SphereSurfaceTest

        surface = SphereSurface()
        #surface = SphereSurfaceTest()
        #surface = EllipsoidSurface()
        #surface = ScaledSurface(surface,scale=[9,3,1])
        mesh = surface.init_mesh()

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        lmesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
        NC = lmesh.number_of_cells()
        a = lmesh.cell_area()
        lmesh.to_vtk(fname=fname)

    def surface_area(self, p=2):
        from fealpy.geometry import SphereSurface

        surface = SphereSurface()
        mesh = surface.init_mesh()
        e = 0
        maxit = 5
        for i in range(maxit):
            node = mesh.entity('node')
            cell = mesh.entity('cell')

            lmesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
            NC = lmesh.number_of_cells()
            a = lmesh.cell_area()
            a = sum(a)
            a_e = (4*np.pi)
            e_new = abs(a-a_e)
            order = np.log2(e/e_new)
            e = e_new
            print("e:", e)
            print("0:", order)
            if i < maxit - 1:
                mesh.uniform_refine(surface = surface)
    
    def jacobi_TMOP_test(self,p=1, plot=True, plotL=True):
        
        #'''
        node = np.array([[0,0],[1,0],[0,1]], dtype = np.float)
        cell = np.array([[0,1,2]],dtype = np.int)

        mesh = TriangleMesh(node,cell)
        '''
        mf = MeshFactory()

        mesh = mf.boxmesh2d([0, 1, 0, 1], nx =2, ny=2, meshtype='tri')
        #node = mesh.entity('node')
        node[0] = [0,1,0]
        node[-1] = [0.8,0]
        node[4] = [0.6,0.6]
        cell = mesh.entity('cell')
        #'''
        ltmesh = LagrangeTriangleMesh(node, cell, p=p)
        NN = ltmesh.number_of_nodes()

        mesh.ds.edge = ltmesh.ds.edge
        mesh.ds.edge2cell = ltmesh.ds.edge2cell

        node = ltmesh.entity('node')
    

        J, L, Q, Delta, S, U, V = ltmesh.jacobi_TMOP()
        a_L = L.sum(axis=-2)/3

        '''
        print('J:')
        print(J)
        print('L:')
        print(L)
        
        print('V:')
        print(V)
        a_V = V[...,0]
        print(a_V)
        print(a_V[:,0])
        print(a_V[:,1])
        print(a_V[:,idx])
        
        print('Q:')
        print(Q)
        print('Delta:')
        print(Delta)
        print('S:')
        print(S)
        print('U:')
        print(U)
        print('np.dot(V,U)')
        print(np.einsum('...ij,...jk',V,U))
        '''
        
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, node=node, showindex=True, fontsize=28)
            #mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, node=node, showindex=True,
                    fontsize=28)
            #mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True,multiindex = a_L)
            plt.show()

       
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
elif sys.argv[1] == 'surface_area':
    p = int(sys.argv[2])
    test.surface_area(p=p)
elif sys.argv[1] == 'jacobi_TMOP_test':
    p = int(sys.argv[2])
    test.jacobi_TMOP_test(p=p)
elif sys.argv[1] =='lineTest':
    p = int(sys.argv[2])
    test.lineTest(p=p)

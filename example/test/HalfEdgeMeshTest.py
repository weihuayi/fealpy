#!/usr/bin/env python3
# 
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from fealpy.mesh import HalfEdgeMesh,Quadtree
from fealpy.mesh import TriangleMesh, PolygonMesh, QuadrangleMesh


class HalfEdgeMeshTest:
    def __init__(self):
        pass

    def refine_poly_test(self, plot=True):
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgeMesh.from_mesh(mesh)

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        clevel = mesh.celldata['level']
        print(clevel)
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        if True:
            isMarkedCell = mesh.mark_helper([2])
            mesh.refine_poly(isMarkedCell, dflag=False)

        if False:
            isMarkedCell = mesh.mark_helper([6])
            mesh.refine_poly(isMarkedCell, dflag=False)

        if False:
            isMarkedCell = mesh.mark_helper([3])
            mesh.refine_poly(isMarkedCell, dflag=False)

        if 1:
            isMarkedCell = mesh.mark_helper([1, 5])
            mesh.refine_poly(isMarkedCell, dflag=False)

        if False:
            isMarkedCell = mesh.mark_helper([1, 12])
            mesh.refine_poly(isMarkedCell, dflag=False)

        if False:
            isMarkedCell = mesh.mark_helper([0, 21])
            mesh.refine_poly(isMarkedCell, dflag=False)
        clevel = mesh.celldata['level']
        print(clevel)
 

        #print("halfedge level:\n")
        #for i, val in enumerate(mesh.halfedgedata['level']):
        #    print(i, ':', val, mesh.ds.halfedge[i, 0:2])

        #print("cell level:\n")
        #for i, val in enumerate(mesh.celldata['level']):
        #    print(i, ':', val)

        if plot:

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)

            NAC = mesh.number_of_all_cells() # 包括外部区域和洞
            cindex = range(mesh.ds.cellstart, NAC)
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.add_halfedge_plot(axes, showindex=True)
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

    def coarsen_poly_test(self, mesh, plot=True):

        if False:
            isMarkedCell = mesh.mark_helper([26, 27, 28, 29, 17, 18, 19, 20, 2, 3])
            mesh.coarsen_poly(isMarkedCell)

        if False:
            isMarkedCell = mesh.mark_helper(
                    [16, 17, 18, 23, 14, 10, 12, 15, 0, 1])
            mesh.coarsen_poly(isMarkedCell)

        if True:
            isMarkedCell = mesh.mark_helper(
                    [7,9,11,13])
            mesh.coarsen_poly(isMarkedCell)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()


        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)

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

    def adaptive_poly_test(self, plot=True):

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
        mesh = HalfEdgeMesh.from_mesh(mesh)

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)

        NE = mesh.number_of_edges()
        nC = mesh.number_of_cells()

        """
        refined mesh
        """
        aopts = mesh.adaptive_options(method='numrefine',maxcoarsen=3,HB=True)
        eta = [0,0,1,1,1]

        mesh.adaptive(eta, aopts)
        print('r',aopts['HB'])

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        plt.show()

        mesh.from_mesh(mesh)
        """
        coarsened mesh
        """
        eta = [0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1]
        #eta = [0,0,0,0,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2]

        mesh.adaptive(eta, aopts)

        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        plt.show()

        if plot:

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)

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


    def triangle_mesh_test(self, plot=False):

        node = np.array([
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
        ])
        cell = np.array([
            (1, 2, 0), (3, 0, 2)
        ])

        tmesh = TriangleMesh(node, cell)
        tmesh.uniform_refine(n=1)
        mesh = HalfEdgeMesh.from_mesh(tmesh)
        if plot:
            halfedge = mesh.ds.halfedge
            for i, idx in enumerate(halfedge):
                print(i, ": " , idx)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def refine_tri_test(self, plot=True):
        node = np.array([
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
        ])
        cell = np.array([
            (1, 2, 0), (3, 0, 2)
        ])
        tmesh = TriangleMesh(node, cell)
        tmesh.uniform_refine(n=1)
        mesh = HalfEdgeMesh.from_mesh(tmesh)
        if plot:
            halfedge = mesh.ds.halfedge
            for i, idx in enumerate(halfedge):
                print(i, ": " , idx)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def refine_quad(self, plot=True):
        cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        mesh = QuadrangleMesh(node, cell)
        #mesh.uniform_refine()
        mesh = HalfEdgeMesh.from_mesh(mesh)
        c = np.array([0.8,0.8])
        r = 0.5
        h = 1e-2
        l=6
        k=0
        NB = 0
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
            mesh.refine_quad(markedcell1)
            k+=1
            print('循环',k,'次***************************')
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            nindex = mesh.nodedata['level']
            mesh.add_plot(axes)
            #mesh.add_halfedge_plot(axes, showindex=True)
            #mesh.find_node(axes, showindex=True, multiindex=nindex)
            plt.show()
        if 0:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)

            cindex, = np.nonzero(mesh.ds.cflag)
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True, multiindex=cindex)

            NN = mesh.number_of_nodes()
            nindex = np.zeros(NN, dtype=np.int)
            halfedge = mesh.ds.halfedge
            nindex = mesh.nodedata['level']
            cindex = mesh.get_data('cell', 'level')
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True, multiindex=nindex)
            mesh.find_cell(axes, showindex=True, multiindex=cindex)
            plt.show()

    def quadtree_test(self, plot=True):
        cell = np.array([[0,1,2,3],[1,4,5,2]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        mesh = Quadtree(node, cell)
        pmesh = mesh.to_pmesh()

        fig = plt.figure()
        axes = fig.gca()
        pmesh.add_plot(axes)
        pmesh.find_node(axes, showindex=True)
        pmesh.find_cell(axes, showindex=True)

        NE = mesh.number_of_edges()
        nC = mesh.number_of_cells()

        aopts =mesh.adaptive_options(method='numrefine',maxcoarsen=3,maxsize=10,HB=True)
        #eta = 2*np.ones(nC,dtype=int)
        eta = [1,2]

        mesh.adaptive(eta, aopts)
        pmesh = mesh.to_pmesh()

        fig = plt.figure()
        axes = fig.gca()
        pmesh.add_plot(axes)
        #pmesh.find_node(axes, showindex=True)
        pmesh.find_cell(axes, showindex=True)
        plt.show()
        print('refine:', aopts['HB'])

        eta = np.zeros(20,np.int)
        eta[4] = -1
        eta[5] = -1
        eta[6] = -1
        eta[7] = -1
        mesh.adaptive(eta, aopts)
        pmesh = mesh.to_pmesh()

        fig = plt.figure()
        axes = fig.gca()
        pmesh.add_plot(axes)
        #pmesh.find_node(axes, showindex=True)
        pmesh.find_cell(axes, showindex=True)
        print('coarsen:', aopts['HB'])

        plt.show()


    def voronoi_test(self, plot=False):
        from scipy.spatial import Delaunay
        from scipy.spatial import Voronoi, voronoi_plot_2d
        from scipy.spatial import KDTree

        points = np.random.rand(10, 2)
        print(points)

        # 边界点固定标记, 在网格生成与自适应算法中不能移除
        # 1: 固定
        # 0: 自由
        fflag = np.ones(4, dtype=np.bool_)

        #  dflag 单元所处的子区域的标记编号
        #  0: 表示外部无界区域
        # -n: n >= 1, 表示编号为 -n 洞
        #  n: n >= 1, 表示编号为  n 的内部子区域
        dflag = np.array([1, 0])

        node = np.array([
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
        halfedge = np.array([
            (1, 0, 1, 3, 4, 1),
            (2, 0, 2, 0, 5, 1),
            (3, 0, 3, 1, 6, 1),
            (0, 0, 0, 2, 7, 1),
            (0, 1, 7, 5, 0, 0),
            (1, 1, 4, 6, 1, 0),
            (2, 1, 5, 7, 2, 0),
            (3, 1, 6, 4, 3, 0)], dtype=np.int)
        NC = 1
        mesh = HalfEdgeMesh(node, halfedge, NC)
        mesh.set_data('fflag', fflag, 'node')
        mesh.set_data('dflag', dflag, 'cell')

        v = Voronoi(points)
        tree = KDTree(points)

        print("points:\n", v.points)
        print('vertices:\n', v.vertices)
        print('ridge_points:\n', v.ridge_points)
        print('ridge_vertices:\n', v.ridge_vertices)
        print('regions:\n', v.regions)
        print('point_region:\n', v.point_region)

        d, nidx = tree.query(node)
        print(nidx)

        if plot:
            halfedge = mesh.ds.halfedge
            for i, idx in enumerate(halfedge):
                print(i, ": " , idx)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            # mesh.find_node(axes, showindex=True, multiindex=nidx)
            mesh.find_node(axes, node=points, color='r', showindex=True)
            mesh.find_node(axes, node=v.vertices, color='b', showindex=True)
            voronoi_plot_2d(v, axes)

            fig = plt.figure()
            axes = fig.gca()
            mesh.add_halfedge_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()


    def refine_triangle_rbTest(self, l, plot=True, rb=True):
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        mesh = TriangleMesh(node, cell)
        #mesh.uniform_refine()
        mesh = HalfEdgeMesh.from_mesh(mesh)
        mesh.ds.cell2hedge = np.array([0, 3, 2, 11, 10])
        c = np.array([0.2,0.2])
        r = 1.2
        h = 1e-2
        k=0
        NB = 0
        start = time.time()
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
            if rb:
                mesh.refine_triangle_rb(markedcell1)
            else:
                mesh.refine_triangle_rbg(markedcell1)
            k+=1
            print('循环',k,'次***************************')
            #print('node', node)
            #print('cell',cell)
        end = time.time()
        print(end-start)
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            nindex = mesh.nodedata['level']
            mesh.add_plot(axes)
            #mesh.add_halfedge_plot(axes, showindex=True)
            #mesh.find_node(axes, showindex=True, multiindex=nindex)
            #mesh.find_cell(axes, showindex=True)
            plt.show()
        if 0:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)

            cindex = np.arange(mesh.number_of_cells()-1)+1
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True, multiindex=cindex)

            NN = mesh.number_of_nodes()
            nindex = np.zeros(NN, dtype=np.int)
            halfedge = mesh.ds.halfedge
            nindex = mesh.nodedata['level']
            cindex = mesh.get_data('cell', 'level')
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True, multiindex=nindex)
            mesh.find_cell(axes, showindex=True, multiindex=cindex)
            plt.show()




test = HalfEdgeMeshTest()
#test.refine_triangle_rbTest(8, plot=True, rb=True)

if sys.argv[1] == 'refine_tri_rb':
    test.refine_triangle_rbTest(int(sys.argv[2]), plot=True, rb=1)

if sys.argv[1] == 'refine_poly':
    test.refine_poly_test(plot=True)

if sys.argv[1] == 'coarsen_poly':
    mesh = test.refine_poly_test(plot=False)

#    fig = plt.figure()
#    axes = fig.gca()
#    mesh.add_plot(axes)
#    mesh.find_node(axes, showindex=True)
#    mesh.find_cell(axes, showindex=True)
#    plt.show()

    test.coarsen_poly_test(mesh, plot=False)

if sys.argv[1] == 'advance_trimesh':
    test.advance_trimesh_test()

if sys.argv[1] == 'adaptive_poly':
    mesh = test.adaptive_poly_test(plot=True)

if sys.argv[1] == 'quadtree_test':
    mesh = test.quadtree_test()

if sys.argv[1] == 'refine_quad':
    test.refine_quad(plot=True)


#test.triangle_mesh_test(plot=True)
#test.voronoi_test(plot=True)

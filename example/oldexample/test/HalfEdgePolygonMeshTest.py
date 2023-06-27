#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import PolygonMesh, HalfEdgePolygonMesh


class HalfEdgePolygonMeshTest:

    def __init__(self):
        pass

    def boundary_edge_to_edge_test(self, plot=True):

        node = np.array([
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        mesh.ds.boundary_edge_to_edge()
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def from_polygonmesh_test(self, plot=True):
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        a = mesh.entity_measure('cell')

        mesh = HalfEdgePolygonMesh.from_polygonmesh(mesh)
        a = mesh.entity_measure('cell')
        mesh.print()

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()


    def refine_test(self, plot=True):
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgePolygonMesh.from_polygonmesh(mesh)
        isMarkedCell = np.zeros(5, dtype=np.bool_)
        isMarkedCell[-1] = True
        isMarkedCell[-2] = True
        mesh.refine(isMarkedCell)


        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[2] = True
            mesh.refine(isMarkedCell)


        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[1] = True
            mesh.refine(isMarkedCell)

        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[13] = True
            mesh.refine(isMarkedCell)

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def edge_to_cell_test(self, plot=True):
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgePolygonMesh.from_polygonmesh(mesh)

        NE = mesh.number_of_edges()
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        print('edge:')
        for i in range(NE):
            print(i, ":", edge[i], edge2cell[i])

        NC = mesh.number_of_cells()
        cell, cellLocation = mesh.entity('cell')
        cell2edge = mesh.ds.cell_to_edge()
        for i in range(NC):
            print(i, ":", cell[cellLocation[i]:cellLocation[i+1]])
            print(i, ":", cell2edge[cellLocation[i]:cellLocation[i+1]])

        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def cell_barycenter_test(self, plot=True):

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
        mesh = HalfEdgePolygonMesh(node, halfedge, NC)
        print(mesh.cell_barycenter())
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def refine_with_flag_test(self, plot=True):
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgePolygonMesh.from_polygonmesh(mesh)

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        name = 'rflag'
        val = np.zeros(2*NE, dtype=np.int)
        mesh.set_data(name, val, 'halfedge')
        
        isMarkedCell = np.zeros(NC+1, dtype=np.bool_)
        isMarkedCell[2] = True
        mesh.refine_with_flag(isMarkedCell, rflag=name, dflag=False)
        
        NC = mesh.number_of_cells()
        isMarkedCell = np.zeros(NC+1, dtype=np.bool_)
        isMarkedCell[6] = True
        mesh.refine_with_flag(isMarkedCell, rflag=name, dflag=False)
        

            
        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC+1, dtype=np.bool_)
            isMarkedCell[3] = True
            mesh.refine_with_flag(isMarkedCell, rflag=name, dflag=False)
            
        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC+1, dtype=np.bool_)
            isMarkedCell[1] = True
            mesh.refine_with_flag(isMarkedCell, rflag=name, dflag=False)
            
        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC+1, dtype=np.bool_)
            isMarkedCell[5] = True
            isMarkedCell[13] = True
            mesh.refine_with_flag(isMarkedCell, rflag=name, dflag=False)
        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC+1, dtype=np.bool_)
            isMarkedCell[0] = True
            isMarkedCell[1] = True
            mesh.refine_with_flag(isMarkedCell, rflag=name, dflag=False)
            
        print("rflag:\n")
        for i, val in enumerate(mesh.halfedgedata[name]):
            print(i, ':', val, mesh.ds.halfedge[i, 0:2])
        
        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            #mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

test = HalfEdgePolygonMeshTest()
#test.boundary_edge_to_edge_test()
#test.from_polygonmesh_test()
#test.refine_test()
#test.edge_to_cell_test()
#test.cell_barycenter_test(plot=False)
test.refine_with_flag_test()

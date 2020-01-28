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
        isMarkedCell = np.zeros(5, dtype=np.bool)
        isMarkedCell[-1] = True
        isMarkedCell[-2] = True
        mesh.refine(isMarkedCell)


        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool)
            isMarkedCell[2] = True
            mesh.refine(isMarkedCell)


        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool)
            isMarkedCell[1] = True
            mesh.refine(isMarkedCell)

        if True:
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool)
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


test = HalfEdgePolygonMeshTest()
#test.boundary_edge_to_edge_test()
#test.from_polygonmesh_test()
test.refine_test()

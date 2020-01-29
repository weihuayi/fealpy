#!/usr/bin/env python3
#
import numpy as np

from fealpy.tools import MatlabShow
from fealpy.mesh import PolygonMesh
from fealpy.mesh import HalfEdgePolygonMesh

class MatlabShowTest():

    def __init__(self):
        pass

    def show_solution_test(self):
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)/2.0
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)

        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgePolygonMesh.from_polygonmesh(mesh)

        def u(p):
            pi = np.pi
            x = p[..., 0]
            y = p[..., 1]
            return np.sin(pi*x)*np.sin(pi*y)

        node = mesh.entity('node')
        solution = u(node)

        plot = MatlabShow()
        plot.show_solution(mesh, solution)


test = MatlabShowTest()
test.show_solution_test()

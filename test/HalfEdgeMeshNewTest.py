#!/usr/bin/env python3
# 
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from fealpy.mesh import HalfEdgeMesh2d
from fealpy.mesh import TriangleMesh, PolygonMesh, QuadrangleMesh


class HalfEdgeMeshNewTest:
    def __init__(self):
        pass

    def HalfEdgeMeshStructure_test(self):
        cell = np.array([[0,1,2],[0,2,3],[1,4,5],[2,1,5]],dtype = np.int)
        node = np.array([[0,0],[1,0],[1,1],[0,1],[2,0],[2,1]], dtype = np.float)
        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        halfedge = mesh.ds.halfedge
        print(halfedge)
        print(mesh.ds.hcell)
        print(mesh.ds.hedge)


test = HalfEdgeMeshNewTest()
test.HalfEdgeMeshStructure_test()



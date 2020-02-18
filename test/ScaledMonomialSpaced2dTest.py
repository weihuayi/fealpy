#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve

from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh import PolygonMesh, QuadrangleMesh
from fealpy.pde.poisson_2d import CosCosData

class ScaledMonomialSpace2dTest:
    def __init__(self):
        pass

    def index1_test(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = ScaledMonomialSpace2d(mesh, 1)
        idx = space.index1(p=p)
        print("p=", p, "\n", idx)

    def index2_test(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = ScaledMonomialSpace2d(mesh, 3)
        idx = space.index2(p=p)
        print("p=", p, "\n", idx)

    def edge_mass_matrix_test(self, p=2):

        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = ScaledMonomialSpace2d(mesh, 3)
        print("new: p=", p, "\n", space.edge_mass_matrix(p=p))
        print("old: p=", p, "\n", space.edge_mass_matrix_1(p=p))

    def interpolation_test(self):
        pass


test = ScaledMonomialSpace2dTest()
test.index1_test(p=1)
test.index1_test(p=2)
test.index1_test(p=3)

test.index2_test(p=3)
test.index2_test(p=4)

test.edge_mass_matrix_test(p=3)

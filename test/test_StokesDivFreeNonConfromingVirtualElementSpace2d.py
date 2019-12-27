import numpy as np

from fealpy.functionspace import StokesDivFreeNonConformingVirtualElementSpace2d
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh import PolygonMesh
from fealpy.pde.poisson_2d import CosCosData

class StokesDivFreeNonConformingVirtualElementSpace2dTest:

    def __init__(self, p=2, h=0.2):
        self.pde = CosCosData()

    def test_index1(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        idx = space.index1(p=3)
        print(idx)

    def test_index2(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        idx = space.index2(p=3)
        print(idx)

    def test_matrix(self, p=2):
        """
        node = np.array([
            (-1.0, -1.0),
            ( 0.0, -1.0),
            ( 1.0, -1.0),
            (-1.0, 0.0),
            ( 1.0, 0.0),
            (-1.0, 1.0),
            ( 0.0, 1.0),
            ( 1.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 4, 7, 6, 5, 3], dtype=np.int)
        cellLocation = np.array([0, 8], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        """

        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        print("G:", space.G)
        print("B:", space.B)
        print("R:", space.R)
        print("J:", space.J)
        print("Q:", space.Q)
        print("L:", space.L)
        print("D:", space.D)

    def test_matrix_A(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p)
        A = space.matrix_A()
        print(A)




test = StokesDivFreeNonConformingVirtualElementSpace2dTest()
#test.test_index1()
#test.test_index2()
#test.test_matrix(p=2)
test.test_matrix_A()

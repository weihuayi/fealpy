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

    def test_matrix(self):
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
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p=3)
        print("G:", self.G)
        print("B:", self.B)
        print("R:", self.R)
        print("J:", self.J)

        print("Q:", self.Q)
        print("L:", self.L)

        print("D:", self.D)




test = StokesDivFreeNonConformingVirtualElementSpace2dTest()
#test.test_index1()
#test.test_index2()
test.test_matrix()

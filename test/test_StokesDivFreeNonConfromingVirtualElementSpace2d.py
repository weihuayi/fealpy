import numpy as np

from fealpy.functionspace import StokesDivFreeNonConformingVirtualElementSpace2d
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh import PolygonMesh
from fealpy.pde.poisson_2d import CosCosData

class StokesDivFreeNonConformingVirtualElementSpace2dTest:

    def __init__(self, p=2, h=0.2):
        self.pde = CosCosData()
        self.mesh = triangle(self.pde.domain(), h, meshtype='polygon')
        self.space = StokesDivFreeNonConformingVirtualElementSpace2d(self.mesh, p)

    def test_index0(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(self.mesh, p)
        idx = space.index0(p=3)
        print(idx)

    def test_index1(self, p=2):
        node = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)], dtype=np.float)
        cell = np.array([0, 1, 2, 3], dtype=np.int)
        cellLocation = np.array([0, 4], dtype=np.int)
        mesh = PolygonMesh(node, cell, cellLocation)
        space = StokesDivFreeNonConformingVirtualElementSpace2d(self.mesh, p)
        idx = space.index1(p=3)
        print(idx)

    def test_matrix_G(self):
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
        space = StokesDivFreeNonConformingVirtualElementSpace2d(mesh, p=2)
        G = space.matrix_G_B()
        print(G)




test = StokesDivFreeNonConformingVirtualElementSpace2dTest()
test.test_index0()
test.test_index1()
#test.test_matrix_G()

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from scipy.sparse import triu, tril, find, hstack
from .mesh_tools import unique_row
from .Mesh3d import Mesh3d, Mesh3dDataStructure 

class HexahedronMeshDataStructure(Mesh3dDataStructure):

    # The following local data structure should be class properties
    localEdge = np.array([
        (0, 1), (1, 2), (2, 3), (0, 3),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (4, 7)])
    localFace = np.array([
        (0, 3, 2, 1), (4, 5, 6, 7), # bottom and top faces
        (0, 4, 7, 3), (1, 2, 6, 5), # left and right faces  
        (0, 1, 5, 4), (2, 3, 7, 6)])# front and back faces
    localFace2edge = np.array([
        (3,  2, 1, 0), (8, 9, 10, 11),
        (4, 11, 7, 3), (1, 6,  9,  5),
        (0,  5, 8, 4), (2, 7, 10,  6)])
    NVC = 8
    NEC = 12
    NFC = 6
    NVF = 4
    NEF = 4

    def __init__(self, NN, cell):
        super(HexahedronMeshDataStructure, self).__init__(NN, cell)

        
    def face_to_edge_sign(self):
        face2edge = self.face_to_edge()
        edge = self.edge
        face2edgeSign = np.zeros((NF, 4), dtype=np.bool)
        for i in range(4):
            face2edgeSign[:, i] = (face[:, i] == edge[face2edge[:, i], 0])
        return face2edgeSign

class HexahedronMesh(Mesh3d):

    def __init__(self, node, cell):
        self.node = node
        NN = node.shape[0]
        self.ds = HexahedronMeshDataStructure(NN, cell)

        self.meshtype = 'hex'

        self.itype = cell.dtype
        self.ftype = node.dtype

    def volume(self):
        pass

    def face_area(self):
        pass

    def jacobi_at_corner(self):
        pass

    def cell_quality(self):
        pass

    def print(self):
        print("Point:\n", self.node)
        print("Cell:\n", self.ds.cell)
        print("Edge:\n", self.ds.edge)
        print("Face:\n", self.ds.face)
        print("Face2cell:\n", self.ds.face2cell)



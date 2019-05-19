import numpy as np
from .Mesh3d import Mesh3d, Mesh3dDataStructure


class HexMeshDataStructure(Mesh3dDataStructure):

    # The following local data structure should be class properties
    localEdge = np.array([
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7)])
    localFace = np.array([
        (0, 2, 4, 6), (1, 3, 5, 7),  # left and right faces
        (0, 1, 4, 5), (2, 3, 6, 7),  # front and back faces
        (0, 1, 2, 3), (4, 5, 6, 7)])  # bottom and top faces

    localFace2edge = np.array([
        (3,  2, 1, 0), (8, 9, 10, 11),
        (4, 11, 7, 3), (1, 6,  9,  5),
        (0,  5, 8, 4), (2, 7, 10,  6)])
    V = 8
    E = 12
    F = 6

    def __init__(self, NN, cell):
        super(HexMeshDataStructure, self).__init__(NN, cell)

    def face_to_edge_sign(self):
        face2edge = self.face_to_edge()
        edge = self.edge
        face = self.face
        NF = len(face2edge)
        face2edgeSign = np.zeros((NF, 4), dtype=np.bool)
        for i in range(4):
            face2edgeSign[:, i] = (face[:, i] == edge[face2edge[:, i], 0])
        return face2edgeSign


class HexMesh(Mesh3d):

    def __init__(self, node, cell):
        self.node = node
        NN = node.shape[0]
        self.ds = HexMeshDataStructure(NN, cell)

        self.meshtype = 'hex'

        self.itype = cell.dtype
        self.ftype = node.dtype

    def disp(self):
        print("Node:\n", self.node)
        print("Cell:\n", self.ds.cell)
        print("Edge:\n", self.ds.edge)
        print("Face:\n", self.ds.face)
        print("Face2cell:\n", self.ds.face2cell)



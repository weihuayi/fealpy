import numpy as np
from .Mesh3d import Mesh3d, Mesh3dDataStructure


class OctreeMeshDataStructure(Mesh3dDataStructure):

    """
    the ordering of corners of a octant, c_zc_yc_x
    0: 000
    1: 001
    2: 010
    3: 011
    4: 100
    5: 101
    6: 110
    7: 111

    NC<===>K, k represent the k-th cube, \phi_k

    cell2cell <===> NO(k, f) = k'
    r \in {0, 1, 2, 3}
    f' \in {0, 1, 2, 3, 4, 5}
    NF(k, f) = 6r + f'

    if f is a boundary,  NF(k, f) = 6r+f

    for k-th octree, is f < f'

    """
    # The following local data structure should be class properties
    localEdge = np.array([
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7)])

    # The corners of a face are enumerated in the same sequence as they
    # occur in the corner numbering of a octant
    localFace = np.array([
        (0, 2, 4, 6), (1, 3, 5, 7),  # left and right faces
        (0, 1, 4, 5), (2, 3, 6, 7),  # front and back faces
        (0, 1, 2, 3), (4, 5, 6, 7)])  # bottom and top faces

    # The adjacent face and face corner of a edge  
    localEdge2Face = np.array([
        (2, 4), (3, 4), (2, 5), (3, 5),
        (0, 4), (1, 4), (0, 5), (1, 5),
        (0, 2), (1, 2), (0, 3), (1, 3)])

    localEdge2FaceCorner = np.array([
        (0, 1, 0, 1),
        (0, 1, 2, 3),
        (2, 3, 0, 1),
        (2, 3, 2, 3),
        (0, 1, 0, 2),
        (0, 1, 1, 3),
        (2, 3, 0, 2),
        (2, 3, 1, 3),
        (0, 2, 0, 2),
        (0, 2, 1, 3),
        (1, 3, 0, 2),
        (1, 3, 1, 3)])

    localFace2Edge = np.array([
        (8, 10, 4, 6),
        (9, 11, 5, 7),
        (8, 9, 0, 2),
        (10, 11, 1, 3),
        (4, 5, 0, 1),
        (6, 7, 2, 3)])

    R = np.array([
        (0, 1, 1, 0, 0, 1),
        (2, 0, 0, 1, 1, 0),
        (2, 0, 0, 1, 1, 0),
        (0, 2, 2, 0, 0, 1),
        (0, 2, 2, 0, 0, 1),
        (2, 0, 0, 2, 2, 0)])
    Q = np.array([
        (1, 2, 5, 6),
        (0, 3, 4, 7),
        (0, 4, 3, 7)])
    P = np.array([
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (1, 0, 3, 2),
        (1, 3, 0, 2),
        (2, 0, 3, 1),
        (2, 3, 0, 1),
        (3, 1, 2, 0),
        (3, 2, 1, 0)])
    V = 8
    E = 12
    F = 6

    def __init__(self, NN, cell):
        super(OctreeMeshDataStructure, self).__init__(NN, cell)


class OctreeMesh(Mesh3d):

    def __init__(self, node, cell):
        self.node = node
        NN = node.shape[0]
        self.ds = OctreeMeshDataStructure(NN, cell)

        self.meshtype = 'octreemesh'

        self.itype = cell.dtype
        self.ftype = node.dtype

    def disp(self):
        print("Node:\n", self.node)
        print("Cell:\n", self.ds.cell)
        print("Edge:\n", self.ds.edge)
        print("Face:\n", self.ds.face)
        print("Face2cell:\n", self.ds.face2cell)



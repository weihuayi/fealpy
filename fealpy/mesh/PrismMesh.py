
import numpy as np
from .Mesh3d import Mesh3d, Mesh3dDataStructure
from ..quadrature import PrismQuadrature


class PrismMesh(Mesh3d):
    def __init__(self, node, cell):
        self.node = node
        self.cell = cell
        NN = node.shape[0]
        self.ds = PrismMeshDataStructure(NN, cell)
        self.meshtype = 'prism'

    def integrator(self, k):
        return PrismQuadrature(k)

    def bc_to_point(self, bc):
        node = self.node
        cell = self.ds.cell
        p0 = np.einsum('...j, ijk->...ik', bc[:, 2:], node[cell[:, 0:3]])
        p1 = np.einsum('...j, ijk->...ik', bc[:, 2:], node[cell[:, 3:]])
        p = np.einsum(
                '..., ...ik->...ik',
                bc[:, 0],
                p0) + np.einsum('..., ...ik->...ik', bc[:, 1], p1)
        return p

    def boundary_face(self):
        face = self.entity('face')
        isBdFace = self.ds.boundary_face_flag()
        return face[isBdFace]


class PrismMeshDataStructure(Mesh3dDataStructure):
    localTFace = np.array([(1, 0, 2),  (3, 4, 5)])
    localQFace = np.array([(1, 2, 5, 4),  (0, 3, 5, 2), (0, 1, 4, 3)])
    localEdge = np.array([
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 4), (2, 5),
        (3, 4), (3, 5), (4, 5)])
    localTFace2edge = np.array([(1, 3, 0), (8, 7, 6)])
    localQFace2edge = np.array([(3, 5, 8, 4), (2, 7, 5, 1), (0, 4, 6, 2)])
    V = 6
    E = 9
    F = 5

    def __init__(self, NN, cell):
        super(PrismMeshDataStructure, self).__init__(NN, cell)

    def total_face(self):
        cell = self.cell
        localTFace = self.localTFace
        totalTFace = cell[:, localTFace].reshape(-1, localTFace.shape[1])

        localQFace = self.localQFace
        totalQFace = cell[:, localQFace].reshape(-1, localQFace.shape[1])

        NTF = totalTFace.shape[0]
        NQF = totalQFace.shape[0]
        totalFace = np.zeros((NTF+NQF, 4), dtype=np.int)
        totalFace[:NTF, -1] = -1e9
        totalFace[:NTF, :-1] = totalTFace
        totalFace[NTF:, :] = totalQFace
        return totalFace

    def total_edge(self):
        cell = self.cell
        localEdge = self.localEdge
        totalEdge = cell[:, localEdge].reshape(-1, localEdge.shape[1])
        return totalEdge

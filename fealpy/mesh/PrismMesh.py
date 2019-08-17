
import numpy as np
from .Mesh3d import Mesh3d, Mesh3dDataStructure
from ..quadrature import PrismQuadrature


class PrismMeshDataStructure(Mesh3dDataStructure):
    localFace = np.array([
        (1, 0, 2, 2),  (3, 4, 5, 5),
        (1, 2, 5, 4),  (0, 3, 5, 2),
        (0, 1, 4, 3)])
    localEdge = np.array([
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 4), (2, 5),
        (3, 4), (3, 5), (4, 5)])
    localFace2edge = np.array([
        (1, 3, 0, 0), (8, 7, 6, 6),
        (3, 5, 8, 4), (2, 7, 5, 1),
        (0, 4, 6, 2)])
    V = 6
    E = 9
    F = 5

    def __init__(self, NN, cell):
        super(PrismMeshDataStructure, self).__init__(NN, cell)


class PrismMesh(Mesh3d):
    def __init__(self, node, cell):
        self.node = node
        self.cell = cell
        NN = node.shape[0]
        self.ds = PrismMeshDataStructure(NN, cell)
        self.meshtype = 'prism'

    def number_of_tri_faces(self):
        face = self.ds.face
        return sum(face[:, -2] == face[:, -1])

    def number_of_quad_faces(self):
        face = self.ds.face
        return sum(face[:, -2] != face[:, -1])

    def integrator(self, k):
        return PrismQuadrature(k)

    def vtk_cell_type(self):
        VTK_PENTAGONAL_PRISM = 15
        return VTK_PENTAGONAL_PRISM

    def bc_to_point(self, bc):
        node = self.node
        cell = self.ds.cell
        bc0 = bc[0]
        bc1 = bc[1]
        p0 = np.einsum('...j, ijk->...ik', bc0, node[cell[:, 0:3]])
        p1 = np.einsum('...j, ijk->...ik', bc0, node[cell[:, 3:]])
        p = np.einsum('', bc1, p0) + np.einsum('', bc1, p1)
        return p

    def multi_index_matrix(self):
        p = 2
        ldof = (p+1)*(p+2)//2
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int8)
        multiIndex[:, 2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 1] = idx0 - multiIndex[:, 2]
        multiIndex[:, 0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        return multiIndex

    def uniform_refine(self, n=1, surface=None, returnim=False):
        """"
        uniform refine prism mesh.
        """
        p = 2
        ldof = (p+1)*(p+1)*(p+2)//2
        w1 = np.zeros((p+1, 2), dtype=np.int8)
        w1[:, 0] = np.arange(p, -1, -1)
        w1[:, 1] = w1[-1::-1, 0]
        w2 = self.multi_index_matrix()
        w3 = np.einsum('ij, km->ijkm', w1, w2)

        w = np.zeros((ldof, 6), dtype=np.int8)
        w[:, 0:3] = w3[:, 0, :, :].reshape(-1, 3)
        w[:, 3:] = w3[:, 1, :, :].reshape(-1, 3)

        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            node = self.entity('node')
            cell = self.entity('cell')
            ps = np.einsum('im, km->ikm', cell + (NN + NC), w)
            ps.sort()
            _, i0, j = np.unique(
                    ps.reshape(-1, 6),
                    return_index=True,
                    return_inverse=True,
                    axis=0)
            ps = np.einsum('km, imd->ikd', w/p/p, node[cell]).reshape(-1, 3)
            self.node = ps[i0]

            cell2newNode = j.reshape(-1, ldof)
            cell = np.zeros((8*NC, 6), dtype=np.int)
            cell[0*NC:1*NC] = cell2newNode[:, [0, 1, 2, 6, 7, 8]]
            cell[1*NC:2*NC] = cell2newNode[:, [1, 3, 4, 7, 9, 10]]
            cell[2*NC:3*NC] = cell2newNode[:, [4, 2, 1, 10, 8, 7]]
            cell[3*NC:4*NC] = cell2newNode[:, [2, 4, 5, 8, 10, 11]]

            cell[4*NC:5*NC] = cell2newNode[:, [6, 7, 8, 12, 13, 14]]
            cell[5*NC:6*NC] = cell2newNode[:, [7, 9, 10, 13, 15, 16]]
            cell[6*NC:7*NC] = cell2newNode[:, [7, 10, 8, 13, 16, 14]]
            cell[7*NC:8*NC] = cell2newNode[:, [8, 10, 11, 14, 16, 17]]
            NN = len(i0)
            self.ds.reinit(NN, cell)



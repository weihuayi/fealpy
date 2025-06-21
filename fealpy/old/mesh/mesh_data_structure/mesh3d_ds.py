
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...common import ranges
from .mesh_ds import HomogeneousMeshDS, StructureMeshDS
from .sparse_tool import arr_to_csr


class Mesh3dDataStructure(HomogeneousMeshDS):
    """
    @brief The topology data structure of 3-d homogeneous mesh.\
           This is an abstract class and can not be used directly.
    """
    # Variables
    face2cell: NDArray
    cell2edge: NDArray

    # Constants
    TD: int = 3
    localFace2edge: NDArray
    localEdge2face: NDArray

    def number_of_edges_of_faces(self):
        """
        @brief Return the number of edges in a face, and is only available in 3d mesh.

        This is equal to the length of `localFace2edge` in axis-1.
        """
        return self.localFace2edge.shape[1]


    ### Special Topology APIs for Non-structures ###

    def cell_to_edge(self, return_sparse=False, return_local=False):
        if not return_sparse:
            return self.cell2edge
        else:
            return arr_to_csr(self.cell2edge, self.number_of_edges(),
                              return_local=return_local, dtype=self.itype)

    def cell_to_cell(
            self, return_sparse=False,
            return_boundary=True, return_array=False):
        """ Get the adjacency information of cells
        """
        if return_array:
            return_sparse = False
            return_boundary = False

        NC = self.number_of_cells()
        NF = self.number_of_faces()

        face2cell = self.face2cell
        if (return_sparse is False) and (return_array is False):
            NFC = self.number_of_faces_of_cells()
            cell2cell = np.zeros((NC, NFC), dtype=self.itype)
            cell2cell[face2cell[:, 0], face2cell[:, 2]] = face2cell[:, 1]
            cell2cell[face2cell[:, 1], face2cell[:, 3]] = face2cell[:, 0]
            return cell2cell

        val = np.ones((NF,), dtype=np.bool_)
        if return_boundary:
            cell2cell = coo_matrix(
                    (val, (face2cell[:, 0], face2cell[:, 1])),
                    shape=(NC, NC))
            cell2cell += coo_matrix(
                    (val, (face2cell[:, 1], face2cell[:, 0])),
                    shape=(NC, NC))
            return cell2cell.tocsr()
        else:
            isInFace = (face2cell[:, 0] != face2cell[:, 1])
            cell2cell = coo_matrix(
                    (
                        val[isInFace],
                        (face2cell[isInFace, 0], face2cell[isInFace, 1])
                    ),
                    shape=(NC, NC), dtype=np.bool_)
            cell2cell += coo_matrix(
                    (
                        val[isInFace],
                        (face2cell[isInFace, 1], face2cell[isInFace, 0])
                    ), shape=(NC, NC), dtype=np.bool_)
            cell2cell = cell2cell.tocsr()
            if return_array is False:
                return cell2cell
            else:
                nn = cell2cell.sum(axis=1).reshape(-1)
                _, adj = cell2cell.nonzero()
                adjLocation = np.zeros(NC+1, dtype=np.int32)
                adjLocation[1:] = np.cumsum(nn)
                return adj.astype(np.int32), adjLocation


    ### General Topology APIs ###

    def cell_to_edge_sign(self, cell=None):
        """
        TODO: true 代表相同方向
        """
        if cell==None:
            cell = self.cell
        NC = self.number_of_cells()
        NEC = self.number_of_edges_of_cells()
        cell2edgeSign = np.zeros((NC, NEC), dtype=np.bool_)
        localEdge = self.localEdge
        E = localEdge.shape[0]
        #for i, (j, k) in zip(range(E), localEdge):
        #    cell2edgeSign[:, i] = cell[:, j] < cell[:, k]
        edge = self.edge
        c2e = self.cell_to_edge()
        cell2edgeSign = edge[c2e, 0]==cell[:, localEdge[:, 0]]
        return cell2edgeSign

    def cell_to_face_sign(self):
        """
        """
        NC = self.number_of_cells()
        face2cell = self.face2cell
        NFC = self.number_of_faces_of_cells()
        cell2facesign = np.zeros((NC, NFC), dtype=np.bool_)
        cell2facesign[face2cell[:, 0], face2cell[:, 2]] = True
        return cell2facesign

    def face_to_edge(self, return_sparse=False):
        cell2edge = self.cell2edge
        face2cell = self.face2cell
        localFace2edge = self.localFace2edge
        face2edge = cell2edge[
                face2cell[:, [0]],
                localFace2edge[face2cell[:, 2]]
                ]
        if return_sparse is False:
            return face2edge
        else:
            NF = self.number_of_faces()
            NE = self.number_of_edges()
            NEF = self.number_of_edges_of_faces()
            f2e = csr_matrix(
                    (
                        np.ones(NEF*NF, dtype=np.bool_),
                        (
                            np.repeat(range(NF), NEF),
                            face2edge.flat
                        )
                    ), shape=(NF, NE))
            return f2e

    def face_to_face(self):
        face2edge = self.face_to_edge(return_sparse=True)
        return face2edge*face2edge.T

    def edge_to_edge(self):
        edge2node = self.edge_to_node(return_sparse=True)
        return edge2node*edge2node.T

    def edge_to_face(self):
        NF = self.number_of_faces()
        NE = self.number_of_edges()
        face2edge = self.face_to_edge()
        NEF = self.number_of_edges_of_faces()
        edge2face = csr_matrix(
                (
                    np.ones(NEF*NF, dtype=np.bool_),
                    (
                        face2edge.flat,
                        np.repeat(range(NF), NEF)
                    )
                ), shape=(NE, NF))
        return edge2face

    def edge_to_cell(self, return_localidx=False):
        NC = self.number_of_cells()
        NE = self.number_of_edges()
        cell2edge = self.cell_to_edge()
        NEC = self.number_of_edges_of_cells()

        if return_localidx is False:
            edge2cell = csr_matrix(
                    (
                        np.ones(NEC*NC, dtype=np.bool_),
                        (
                            cell2edge.flat,
                            np.repeat(range(NC), NEC)
                        )
                    ), shape=(NE, NC))
        else:
            raise ValueError("Need to implement!")

        return edge2cell

    def node_to_node(self):
        """ The neighbor information of nodes
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edge = self.edge
        node2node = csr_matrix(
                (
                    np.ones((2*NE,), dtype=np.bool_),
                    (
                        edge.flat,
                        edge[:, [1, 0]].flat
                    )
                ), shape=(NN, NN), dtype=np.bool_)
        return node2node

    def node_to_edge(self):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edge = self.edge
        node2edge = csr_matrix(
                (
                    np.ones(2*NE, dtype=np.bool_),
                    (
                        edge.flat,
                        np.repeat(range(NE), 2)
                    )
                ), shape=(NN, NE))
        return node2edge

    def node_to_face(self):
        NN = self.number_of_nodes()
        NF = self.number_of_faces()

        face = self.face
        NVF = self.number_of_vertices_of_faces()
        node2face = csr_matrix(
                (
                    np.ones(NVF*NF, dtype=np.bool_),
                    (
                        face.flat,
                        np.repeat(range(NF), NVF)
                    )
                ), shape=(NN, NF))
        return node2face

    def node_to_cell(self, return_localidx=False):
        """
        """
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NVC = self.number_of_vertices_of_cells()

        cell = self.cell

        if return_localidx is False:
            node2cell = csr_matrix(
                    (
                        np.ones(NVC*NC, dtype=np.bool_),
                        (
                            cell.flatten(),
                            np.repeat(range(NC), NVC)
                        )
                    ), shape=(NN, NC), dtype=np.bool_)
        else:
            node2cell = csr_matrix(
                    (
                        ranges(NVC*np.ones(NC, dtype=self.itype), start=1),
                        (
                            cell.flatten(),
                            np.repeat(range(NC), NVC)
                        )
                    ), shape=(NN, NC), dtype=self.itype)
        return node2cell


class StructureMesh3dDataStructure(StructureMeshDS, Mesh3dDataStructure):
    cw = np.array([0, 1, 3, 2])
    ccw = np.array([0, 2, 3, 1])
    localEdge = np.array([
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 1), (2, 3), (4, 5), (6, 7)])
    localFace = np.array([
        (0, 1, 2, 3), (4, 5, 6, 7),  # left and right faces
        (0, 1, 4, 5), (2, 3, 6, 7),  # front and back faces
        (0, 2, 4, 6), (1, 3, 5, 7)])  # bottom and top faces
    localFace2edge = np.array([
        (4, 5, 8, 9), (6, 7, 10, 11),
        (0, 1, 8, 10), (2, 3, 9, 11),
        (0, 2, 4, 6), (1, 3, 5, 7)])


    @property
    def face(self):
        """
        @brief 生成网格中所有的面
        """
        NN = self.NN
        NF = self.number_of_faces()

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        face = np.zeros((NF, 4), dtype=np.int_)

        NF0 = 0
        NF1 = (nx + 1) * ny * nz
        c = idx[:, :-1, :-1]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + 1
        face[NF0:NF1, 2] = face[NF0:NF1, 0] + nz + 1
        face[NF0:NF1, 3] = face[NF0:NF1, 2] + 1
        face[NF0:NF0 + ny * nz, :] = face[NF0:NF0 + ny * nz, [1, 0, 3, 2]]

        NF0 = NF1
        NF1 += nx * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:-1, :, :-1]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + 1
        face[NF0:NF1, 2] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        face[NF0:NF1, 3] = face[NF0:NF1, 2] + 1
        NF2 = NF0 + ny * nz
        N = nz * (ny + 1)
        idx1 = np.zeros((nx, nz), dtype=np.int_)
        idx1 = np.arange(NF2, NF2 + nz)
        idx1 = idx1 + np.arange(0, N * nx, N).reshape(nx, 1)
        idx1 = idx1.flatten()
        face[idx1] = face[idx1][:, [1, 0, 3, 2]]

        NF0 = NF1
        NF1 += nx * ny * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:-1, :-1, :]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + nz + 1
        face[NF0:NF1, 2] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        face[NF0:NF1, 3] = face[NF0:NF1, 2] + nz + 1
        N = ny * (nz + 1)
        idx2 = np.zeros((nx, ny), dtype=np.int_)
        idx2 = np.arange(NF0, NF0 + ny * (nz + 1), nz + 1)
        idx2 = idx2 + np.arange(0, N * nx, N).reshape(nx, 1)
        idx2 = idx2.flatten()
        face[idx2] = face[idx2][:, [1, 0, 3, 2]]

        return face


    @property
    def edge(self):
        """
        @brief 生成网格中所有的边
        """
        NN = self.NN
        NE = self.number_of_edges()

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        edge = np.zeros((NE, 2), dtype=np.int_)

        NE0 = 0
        NE1 = nx * (ny + 1) * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:-1, :, :]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + (ny + 1) * (nz + 1)

        NE0 = NE1
        NE1 += (nx + 1) * ny * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:, :-1, :]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + nz + 1

        NE0 = NE1
        NE1 += (nx + 1) * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:, :, :-1]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + 1

        return edge


    @property
    def face2cell(self):
        NF = self.number_of_faces()
        NC = self.number_of_cells()

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NC).reshape(nx, ny, nz)
        face2cell = np.zeros((NF, 4), dtype=np.int_)

        # x direction
        NF0 = 0
        NF1 = (nx+1) * ny * nz
        face2cell[NF0:NF1-ny*nz, 0] = idx.flatten()
        face2cell[NF0+ny*nz:NF1, 1] = idx.flatten()
        face2cell[NF0:NF1-ny*nz, 2] = 0
        face2cell[NF0:NF1-ny*nz, 3] = 1

        face2cell[NF1-ny*nz:NF1, 0] = idx[-1].flatten()
        face2cell[NF0:NF0+ny*nz, 1] = idx[0].flatten()
        face2cell[NF1-ny*nz:NF1, 2] = 1
        face2cell[NF0:NF0+ny*nz, 3] = 0

        # y direction
        idy = np.swapaxes(idx, 1, 0)
        NF0 = NF1
        NF1 += nx * (ny+1) * nz

        fidy = np.arange(NF0, NF1).reshape(nx, ny+1, nz).swapaxes(0, 1)

        face2cell[fidy[:-1], 0] = idy
        face2cell[fidy[1:], 1] = idy
        face2cell[fidy[:-1], 2] = 0
        face2cell[fidy[1:], 3] = 1

        face2cell[fidy[-1], 0] = idy[-1]
        face2cell[fidy[0], 1] = idy[0]
        face2cell[fidy[-1], 2] = 1
        face2cell[fidy[0], 3] = 0

        # z direction
        idz = np.transpose(idx, (2, 0, 1))
        NF0 = NF1
        NF1 += nx * ny * (nz + 1)

        fidz = np.arange(NF0, NF1).reshape(nx, ny, nz+1).transpose(2, 0, 1)

        face2cell[fidz[:-1], 0] = idz
        face2cell[fidz[1:], 1] = idz
        face2cell[fidz[:-1], 2] = 0
        face2cell[fidz[1:], 3] = 1

        face2cell[fidz[-1], 0] = idz[-1]
        face2cell[fidz[0], 1] = idz[0]
        face2cell[fidz[-1], 2] = 1
        face2cell[fidz[0], 3] = 0
        return face2cell

    @property
    def cell2edge(self):
        """
        The neighbor information of cell to edge
        @brief 单元和边的邻接关系, 储存每个单元相邻的边的编号
        """
        NC = self.number_of_cells()

        nx = self.nx
        ny = self.ny
        nz = self.nz

        cell2edge = np.zeros((NC, 12), dtype=np.int_)

        idx0 = np.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
        cell2edge[:, 0] = idx0[:, :-1, :-1].flatten()
        cell2edge[:, 1] = idx0[:, :-1, 1:].flatten()
        cell2edge[:, 2] = idx0[:, 1:, :-1].flatten()
        cell2edge[:, 3] = idx0[:, 1:, 1:].flatten()

        NE0 = nx * (ny + 1) * (nz + 1)
        idx1 = np.arange((nx + 1) * ny * (nz + 1)).reshape(nx + 1, ny, nz + 1)
        cell2edge[:, 4] = (NE0 + idx1[:-1, :, :-1]).flatten()
        cell2edge[:, 5] = (NE0 + idx1[:-1, :, 1:]).flatten()
        cell2edge[:, 6] = (NE0 + idx1[1:, :, :-1]).flatten()
        cell2edge[:, 7] = (NE0 + idx1[1:, :, 1:]).flatten()

        NE1 = nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1)
        idx2 = np.arange((nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny + 1, nz)
        cell2edge[:, 8] = (NE1 + idx2[:-1, :-1, :]).flatten()
        cell2edge[:, 9] = (NE1 + idx2[:-1, 1:, :]).flatten()
        cell2edge[:, 10] = (NE1 + idx2[1:, :-1, :]).flatten()
        cell2edge[:, 11] = (NE1 + idx2[1:, 1:, :]).flatten()

        return cell2edge

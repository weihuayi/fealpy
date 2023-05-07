
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...common import ranges
from .mesh_ds import MeshDataStructure, StructureMeshDS


class Mesh3dDataStructure(MeshDataStructure):
    """
    @brief The topology data structure of 3-d mesh.\
           This is an abstract class and can not be used directly.
    """
    # Variables
    face2cell: NDArray
    cell2edge: NDArray

    # Constants
    TD: int = 3

    def construct(self):
        NC = self.number_of_cells()

        totalFace = self.total_face()

        _, i0, j = np.unique(
                np.sort(totalFace, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)

        self.face = totalFace[i0]

        NF = i0.shape[0]
        self.NF = NF

        self.face2cell = np.zeros((NF, 4), dtype=self.itype)

        i1 = np.zeros(NF, dtype=self.itype)
        NFC = self.number_of_faces_of_cells()
        i1[j] = np.arange(NFC*NC)

        self.face2cell[:, 0] = i0 // NFC
        self.face2cell[:, 1] = i1 // NFC
        self.face2cell[:, 2] = i0 % NFC
        self.face2cell[:, 3] = i1 % NFC

        totalEdge = self.total_edge()
        self.edge, i2, j = np.unique(
                np.sort(totalEdge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NEC = self.number_of_edges_of_cells()
        self.cell2edge = np.reshape(j, (NC, NEC))
        self.NE = self.edge.shape[0]

    def clear(self):
        self.face = None
        self.edge = None
        self.face2cell = None
        self.cell2edge = None


    ### Cell ###

    def cell_to_edge(self, return_sparse=False):
        """ The neighbor information of cell to edge
        """
        if return_sparse is False:
            return self.cell2edge
        else:
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            cell2edge = coo_matrix((NC, NE), dtype=np.bool_)
            NEC = self.number_of_edges_of_cells()
            cell2edge = csr_matrix(
                    (
                        np.ones(NEC*NC, dtype=np.bool_),
                        (
                            np.repeat(range(NC), NEC),
                            self.cell2edge.flat
                        )
                    ), shape=(NC, NE))
            return cell2edge

    def cell_to_edge_sign(self, cell=None):
        """
        TODO: check here
        """
        if cell==None:
            cell = self.cell
        NC = self.number_of_cells()
        NEC = self.number_of_edges_of_cells()
        cell2edgeSign = np.zeros((NC, NEC), dtype=np.bool_)
        localEdge = self.localEdge
        E = localEdge.shape[0]
        for i, (j, k) in zip(range(E), localEdge):
            cell2edgeSign[:, i] = cell[:, j] < cell[:, k]
        return cell2edgeSign

    def cell_to_face(self):
        NC = self.number_of_cells()
        NF = self.number_of_faces()
        face2cell = self.face2cell
        NFC = self.number_of_faces_of_cells()
        cell2face = np.zeros((NC, NFC), dtype=self.itype)
        cell2face[face2cell[:, 0], face2cell[:, 2]] = range(NF)
        cell2face[face2cell[:, 1], face2cell[:, 3]] = range(NF)
        return cell2face

    def cell_to_face_sign(self):
        """
        """
        NC = self.number_of_cells()
        face2cell = self.face2cell
        NFC = self.number_of_faces_of_cells()
        cell2facesign = np.zeros((NC, NFC), dtype=np.bool_)
        cell2facesign[face2cell[:, 0], face2cell[:, 2]] = True
        return cell2facesign

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
            NFC = self.NFC
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


    ### face ###

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
            NEF = self.NEF
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

    def face_to_cell(self, return_sparse=False):
        if return_sparse is False:
            return self.face2cell
        else:
            NC = self.number_of_cells()
            NF = self.number_of_faces()
            face2cell = csr_matrix(
                    (
                        np.ones(2*NF, dtype=np.bool_),
                        (
                            np.repeat(range(NF), 2),
                            self.face2cell[:, [0, 1]].flat
                        )
                    ), shape=(NF, NC), dtype=np.bool_)
            return face2cell


    ### edge ###

    def edge_to_edge(self):
        edge2node = self.edge_to_node(return_sparse=True)
        return edge2node*edge2node.T

    def edge_to_face(self):
        NF = self.number_of_faces()
        NE = self.number_of_edges()
        face2edge = self.face_to_edge()
        NEF = self.NEF
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


    ### Node ###

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
        NVF = self.NVF
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
    pass

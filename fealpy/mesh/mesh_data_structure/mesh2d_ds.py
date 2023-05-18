
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...common import ranges
from .mesh_ds import Redirector, MeshDataStructure, StructureMeshDS


class Mesh2dDataStructure(MeshDataStructure):
    """
    @brief The topology data structure of 2-d mesh.\
           This is an abstract class and can not be used directly.
    """
    # Variables
    face: Redirector[NDArray] = Redirector('edge')
    edge2cell: NDArray

    # Constants
    TD: int = 2

    ### cell ###

    def cell_to_edge(self):
        """
        @brief The neighbor information of cell to edge
        """
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        NEC = self.number_of_edges_of_cells()

        edge2cell = self.edge2cell

        cell2edge = np.zeros((NC, NEC), dtype=self.itype)
        cell2edge[edge2cell[:, 0], edge2cell[:, 2]] = np.arange(NE)
        cell2edge[edge2cell[:, 1], edge2cell[:, 3]] = np.arange(NE)
        return cell2edge

    def cell_to_edge_sign(self):
        NC = self.number_of_cells()
        NEC = self.number_of_edges_of_cells()

        edge2cell = self.edge2cell

        cell2edgeSign = np.zeros((NC, NEC), dtype=np.bool_)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = True

        return cell2edgeSign

    cell_to_face = cell_to_edge
    cell_to_face_sign = cell_to_edge_sign

    def cell_to_cell(self, return_sparse=False, return_boundary=True, return_array=False):
        """ Consctruct the neighbor information of cells
        """
        if return_array:
             return_sparse = False
             return_boundary = False

        NC = self.number_of_cells()
        NE = self.number_of_edges()

        edge2cell = self.edge2cell
        if (return_sparse == False) & (return_array == False):
            NEC = self.NEC
            cell2cell = np.zeros((NC, NEC), dtype=self.itype)
            cell2cell[edge2cell[:, 0], edge2cell[:, 2]] = edge2cell[:, 1]
            cell2cell[edge2cell[:, 1], edge2cell[:, 3]] = edge2cell[:, 0]
            return cell2cell

        val = np.ones((NE,), dtype=np.bool_)
        if return_boundary:
            cell2cell = coo_matrix(
                    (val, (edge2cell[:, 0], edge2cell[:, 1])),
                    shape=(NC, NC), dtype=np.bool_)
            cell2cell += coo_matrix(
                    (val, (edge2cell[:, 1], edge2cell[:, 0])),
                    shape=(NC, NC), dtype=np.bool_)
            return cell2cell.tocsr()
        else:
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            cell2cell = coo_matrix(
                    (val[isInEdge], (edge2cell[isInEdge, 0], edge2cell[isInEdge, 1])),
                    shape=(NC, NC), dtype=np.bool_)
            cell2cell += coo_matrix(
                    (val[isInEdge], (edge2cell[isInEdge, 1], edge2cell[isInEdge, 0])),
                    shape=(NC, NC), dtype=np.bool_)
            cell2cell = cell2cell.tocsr()
            if return_array == False:
                return cell2cell
            else:
                nn = cell2cell.sum(axis=1).reshape(-1)
                _, adj = cell2cell.nonzero()
                adjLocation = np.zeros(NC+1, dtype=np.int32)
                adjLocation[1:] = np.cumsum(nn)
                return adj.astype(np.int32), adjLocation

    ### face ###

    def face_to_cell(self):
        return self.edge2cell

    ### edge ###

    edge_to_cell = face_to_cell

    ### node ###

    def node_to_node(self, return_array=False):

        """
        Notes
        -----
            节点与节点的相邻关系

        TODO
        ----
            曲边元的边包含两个以上的点,
        """

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edge = self.edge
        I = edge[:, [0, -1]].flat
        J = edge[:, [-1, 0]].flat
        val = np.ones((2*NE,), dtype=np.bool_)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN))
        if return_array == False:
            return node2node
        else:
            nn = node2node.sum(axis=1).reshape(-1)
            _, adj = node2node.nonzero()
            adjLocation = np.zeros(NN+1, dtype=np.int32)
            adjLocation[1:] = np.cumsum(nn)
            return adj.astype(np.int32), adjLocation

    def node_to_node_in_edge(self, NN, edge):
        """
        Notes
        ----
        TODO
        """
        I = edge.flatten()
        J = edge[:, [1, 0]].flatten()
        val = np.ones(2*edge.shape[0], dtype=np.bool_)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool_)
        return node2node

    def node_to_edge(self):
        """
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NVE = self.NVE
        I = self.edge.flat
        J = np.repeat(range(NE), NVE)
        val = np.ones(NVE*NE, dtype=np.bool_)
        node2edge = csr_matrix((val, (I, J)), shape=(NN, NE))
        return node2edge

    node_to_face = node_to_edge

    def node_to_cell(self, return_localidx=False):
        """
        """
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NVC = self.number_of_vertices_of_cells()

        I = self.cell.flat
        J = np.repeat(range(NC), NVC)

        if return_localidx == False:
            val = np.ones(NVC*NC, dtype=np.bool_)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC))
        else:
            val = ranges(NVC*np.ones(NC, dtype=self.itype), start=1)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=self.itype)
        return node2cell

    ### boundary ###

    def boundary_edge_to_edge(self):
        """
        """
        NN = self.NN
        edge = self.edge
        index = self.boundary_edge_index()
        bdEdge = edge[index]
        n = bdEdge.shape[0]
        val = np.ones(n, dtype=np.bool_)
        m0 = csr_matrix((val, (range(n), bdEdge[:, 0])), shape=(n, NN))
        m1 = csr_matrix((val, (range(n), bdEdge[:, 1])), shape=(n, NN))
        _, pre = (m0*m1.T).nonzero()
        _, nex = (m1*m0.T).nonzero()
        return index[pre], index[nex]

    def boundary_edge_flag(self):
        return self.boundary_face_flag()


class StructureMesh2dDataStructure(StructureMeshDS, Mesh2dDataStructure):
    pass


import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...common import ranges
from .mesh_ds import ArrRedirector, HomogeneousMeshDS, StructureMeshDS
from .sparse_tool import arr_to_csr

class Mesh2dDataStructure(HomogeneousMeshDS):
    """
    @brief The topology data structure of 2-d homogeneous mesh.\
           This is an abstract class and can not be used directly.
    """
    # Variables
    face = ArrRedirector('edge')
    edge2cell: NDArray

    # Constants
    TD: int = 2


    ### Special Topology APIs for Non-structures ###

    def cell_to_edge(self, return_sparse=False, return_local=False):
        return self.cell_to_face(return_sparse=return_sparse, return_local=return_local)

    def cell_to_cell(self, return_sparse=False, return_boundary=True, return_array=False):
        """ Consctruct the neighbor information of cells
        """
        if return_array:
             return_sparse = False
             return_boundary = False

        NC = self.number_of_cells()
        NE = self.number_of_edges()

        edge2cell = self.edge2cell
        if (return_sparse == False) and (return_array == False):
            NEC = self.number_of_edges_of_cells()
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


    ### General Topology APIs ###

    def cell_to_edge_sign(self):
        NC = self.number_of_cells()
        NEC = self.number_of_edges_of_cells()

        edge2cell = self.edge2cell

        cell2edgeSign = np.zeros((NC, NEC), dtype=np.bool_)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = True

        return cell2edgeSign

    def cell_to_face_sign(self):
        return self.cell_to_edge_sign()

    def edge_to_cell(self, return_sparse=False):
        return self.face_to_cell(return_sparse=return_sparse)

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

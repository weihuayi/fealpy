
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...common import ranges
from .mesh_ds import ArrRedirector, HomogeneousMeshDS, StructureMeshDS


class Mesh1dDataStructure(HomogeneousMeshDS):
    """
    @brief The topology data structure of 1-d mesh.\
           This is an abstract class and can not be used directly.
    """
    # Variables
    edge = ArrRedirector('cell')

    # Constants
    TD = 1
    localEdge = np.array([(0, 1)], dtype=np.int_)
    localFace = np.array([(0, ), (1, )], dtype=np.int_)

    ### Special Topology APIs for Non-structures ###

    def cell_to_face(self, return_sparse=False, return_local=False) -> NDArray: # simplified for 1d case
        return self.cell_to_node(return_sparse=return_sparse, return_local=return_local)

    def cell_to_edge(self) -> NDArray:
        NC = self.number_of_cells()
        return np.arange(NC, dtype=self.itype).reshape(NC, 1)


    ### General Topology APIs ###

    def face_to_edge(self, return_sparse=False):
        return self.face_to_cell(return_sparse=return_sparse)

    def edge_to_face(self, return_sparse=False):
        return self.cell_to_face(return_sparse=return_sparse)

    def edge_to_cell(self, return_sparse=False):
        return self.cell_to_edge(return_sparse=return_sparse)


class StructureMesh1dDataStructure(StructureMeshDS, Mesh1dDataStructure):
    localEdge = np.array([(0, 1)])
    localFace = np.array([(0,), (1,)])
    face = ArrRedirector('_face')

    @property
    def _face(self):
        NN = self.number_of_nodes()
        return np.arange(NN).reshape(NN, 1)

    @property
    def face2cell(self) -> NDArray:
        """
        @TODO
        """
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        assert NC + 1 == NN
        I = np.arange(NC)
        node2cell = np.zeros((NN, 4), dtype=self.itype)
        node2cell[1:, 0] = I
        node2cell[:-1, 1] = I
        node2cell[1:, 2] = 1
        node2cell[:-1, 3] = 0
        node2cell[-1, 1] = NC - 1
        node2cell[-1, 3] = 1
        return node2cell

    # boundary, simplified for 1d structure

    def boundary_node_flag(self):
        """
        @brief Determine boundary nodes in the 1D structure mesh.

        @return isBdNode : np.array, dtype=np.bool_
            An array of booleans where True indicates a boundary node.
        """
        isBdNode = np.zeros(self.NN, dtype=np.bool_)
        isBdNode[0] = True
        isBdNode[-1] = True
        return isBdNode

    def boundary_cell_flag(self):
        """
        @brief Determine boundary cells in the 1D structure mesh.

        @return isBdCell : np.array, dtype=np.bool_
            An array of booleans where True indicates a boundary cell.
        """
        NC = self.number_of_cells()
        isBdCell = np.zeros((NC,), dtype=np.bool_)
        isBdCell[0] = True
        isBdCell[-1] = True
        return isBdCell

    def boundary_node_index(self):
        """
        @brief Get the indices of boundary nodes in the 1D structure mesh.

        @return boundary_node_indices : np.array, dtype=self.itype
            An array containing the indices of the boundary nodes.
        """
        return np.array([0, self.NN-1], dtype=self.itype)

    def boundary_cell_index(self):
        """
        @brief Get the indices of boundary cells in the 1D structure mesh.

        This function returns an array containing the indices of the
        boundary cells in the mesh.

        @return boundary_cell_indices : np.array, dtype=self.itype
            An array containing the indices of the boundary cells.
        """
        return np.array([0, self.NC-1], dtype=self.itype)

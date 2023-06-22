
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

    def cell_to_edge(self) -> NDArray:
        NC = self.number_of_cells()
        return np.arange(NC, dtype=self.itype).reshape(NC, 1)

    def cell_to_face(self) -> NDArray:
        return self.cell

    def face_to_cell(self) -> NDArray:
        return self.face2cell


    ### General Topology APIs ###

    face_to_edge = face_to_cell
    edge_to_face = cell_to_face
    edge_to_cell = cell_to_edge
    node_to_edge = face_to_cell
    node_to_cell = face_to_cell


class StructureMesh1dDataStructure(StructureMeshDS, Mesh1dDataStructure):

    def face_to_cell(self) -> NDArray:
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

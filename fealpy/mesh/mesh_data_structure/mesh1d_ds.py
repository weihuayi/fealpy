
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...common import ranges
from .mesh_ds import Redirector, HomogeneousMeshDS, StructureMeshDS


class Mesh1dDataStructure(HomogeneousMeshDS):
    """
    @brief The topology data structure of 1-d mesh.\
           This is an abstract class and can not be used directly.
    """
    # Variables
    edge: Redirector[NDArray] = Redirector('cell')

    # Constants
    TD = 1
    NEC = 1
    NFC = 2
    localEdge = np.array([(0, 1)], dtype=np.int_)
    localFace = np.array([(0, ), (1, )], dtype=np.int_)

    @property
    def face(self):
        NN = self.number_of_nodes()
        return np.arange(NN, dtype=self.itype).reshape(NN, 1)

    def construct(self) -> None:
        pass

    ### cell ###

    def cell_to_edge(self) -> NDArray:
        NC = self.number_of_cells()
        return np.arange(NC, dtype=self.itype).reshape(NC, 1)

    def cell_to_face(self) -> NDArray:
        return self.cell

    ### face ###

    def face_to_cell(self) -> NDArray:
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

    face_to_edge = face_to_cell

    ### edge ###

    edge_to_face = cell_to_face
    edge_to_cell = cell_to_edge

    ### node ###

    node_to_edge = face_to_cell
    node_to_cell = face_to_cell


class StructureMesh1dDataStructure(StructureMeshDS, Mesh1dDataStructure):
    pass

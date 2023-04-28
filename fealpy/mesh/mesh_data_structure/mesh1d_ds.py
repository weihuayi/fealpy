
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...common import ranges
from .mesh_ds import Redirector, RegularCellMeshDS, StructureMeshDS


class Mesh1dDataStructure(RegularCellMeshDS):
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

    def construct(self) -> None:
        pass

    def cell_to_edge(self) -> NDArray:
        NC = self.number_of_cells()
        return np.arange(NC, dtype=self.itype).reshape(NC, 1)

    def cell_to_face(self) -> NDArray:
        return self.cell

    def face_to_cell(self) -> NDArray:
        NN = self.number_of_nodes()
        I = np.arange(NN)
        node2cell = np.zeros((NN, 4), dtype=self.itype)
        node2cell[:, 0] = I - 1
        node2cell[:, 1] = I
        node2cell[:, 2] = 1
        node2cell[:, 3] = 0
        return node2cell


class StructureMesh1dDataStructure(StructureMeshDS, Mesh1dDataStructure):
    pass

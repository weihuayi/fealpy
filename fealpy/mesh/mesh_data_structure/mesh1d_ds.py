
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from ...common import ranges
from .mesh_ds import MeshDataStructure, Redirector


class Mesh1dDataStructure(MeshDataStructure):
    """
    @brief The topology data structure of 1-d mesh.\
           This is an abstract class and can not be used directly.
    """
    TD = 1
    edge: Redirector[NDArray] = Redirector('cell')

    # TODO

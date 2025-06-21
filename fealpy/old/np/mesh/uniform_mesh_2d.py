from typing import Optional, Union, List

import numpy as np
from numpy.typing import NDArray

from fealpy.np.mesh.mesh_base import _S
from fealpy.np.mesh.quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import HomogeneousMesh, estr2dim

Index = Union[NDArray, int, slice]
_dtype = np.dtype

_S = slice(None)

class UniformMesh2D(HomogeneousMesh):
    def __init__(self, node: NDArray, cell: NDArray) -> None:
        super().__init__(TD=2)

        kwargs = {'dtype': cell.dtype}
        self.cell = cell
        self.localEdge = np.array([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.localFace = np.array([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.ccw = np.array([0, 1, 2], **kwargs)

        self.localCell = np.array([
            (0, 1, 2, 3),
            (1, 2, 3, 0),
            (2, 3, 0, 1),
            (3, 0, 1, 2)], **kwargs)

        self.construct()

        self.node = node
        GD = node.size(-1)
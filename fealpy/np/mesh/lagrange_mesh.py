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

class LagrangeMesh(HomogeneousMesh):
    pass

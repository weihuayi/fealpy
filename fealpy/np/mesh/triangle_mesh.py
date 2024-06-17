from typing import Optional, Union, List

import numpy as np

from fealpy.np.mesh.mesh_base import _S
from fealpy.np.mesh.quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import SimplexMesh, estr2dim

_dtype = np.dtype

_S = slice(None)


class TriangleMesh(SimplexMesh):
    pass
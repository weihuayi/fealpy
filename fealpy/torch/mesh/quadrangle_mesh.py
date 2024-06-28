from typing import Optional, Union, List

import numpy as np
import torch
from torch import Tensor

from .quadrature import Quadrature

from .. import logger
from . import functional as F #TODO: maybe just import nessesary functioals
from .mesh_base import HomogeneousMesh, estr2dim

Index = Union[Tensor, int, slice]
_dtype = torch.dtype
_device = torch.device
_S = slice(None)


class QuadrangleMesh(HomogeneousMesh):
    def __init__(self, node: Tensor, cell: Tensor) -> None:
        super().__init__(TD=2)
        # constant tensors
        kwargs = {'dtype': cell.dtype, 'device': cell.device}
        self.localEdge = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.localFace = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.ccw = torch.tensor([0, 1, 2, 3], **kwargs)

        self.localCell = torch.tensor([
            (0, 1, 2, 3),
            (1, 2, 3, 0),
            (2, 3, 0, 1),
            (3, 0, 1, 2)], **kwargs)

        self.construct()

        self.node = node
        GD = node.size(-1)

        if GD == 2: # TODO: implement functionals here.
            self._cell_area = F.simplex_measure
            self._grad_lambda = F.tri_grad_lambda_2d
        elif GD == 3:
            self._cell_area = F.tri_area_3d
            self._grad_lambda = F.tri_grad_lambda_3d
        else:
            logger.warn(f"{GD}D quadrangle mesh is not well supported: "
                        "cell_area and grad_lambda are not available. "
                        "Any operation involving them will fail.")

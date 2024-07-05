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
            
    # constructor
    @classmethod
    def from_box(cls, box: List[int]=[0, 1, 0, 1], nx=10, ny=10, threshold=None, *,
                 itype: Optional[_dtype]=torch.int,
                 ftype: Optional[_dtype]=torch.float64,
                 device: Union[_device, str, None]=None,
                 require_grad: bool=False):
        """Generate a uniform triangle mesh for a box domain.

        Parameters:
            box (List[int]): 4 integers, the left, right, bottom, top of the box.\n
            nx (int, optional): Number of divisions along the x-axis, defaults to 10.\n
            ny (int, optional): Number of divisions along the y-axis, defaults to 10.\n
            threshold (Callable | None, optional): Optional function to filter cells.
                Based on their barycenter coordinates, defaults to None.

        Returns:
            TriangleMesh: Triangle mesh instance.
        """
        fkwargs = {'dtype': ftype, 'device': device}
        ikwargs = {'dtype': itype, 'device': device}
        NN = (nx + 1) * (ny + 1)
        NC = nx * ny
        X, Y = torch.meshgrid(
            torch.linspace(box[0], box[1], nx + 1, **fkwargs),
            torch.linspace(box[2], box[3], ny + 1, **fkwargs),
            indexing='ij'
        )
        node = torch.stack([X.ravel(), Y.ravel()], dim=-1)

        idx = torch.arange(NN, **ikwargs).reshape(nx + 1, ny + 1)
        cell = torch.zeros((2 * NC, 3), **ikwargs)
        cell[:NC, 0] = idx[1:, 0:-1].T.flatten()
        cell[:NC, 1] = idx[1:, 1:].T.flatten()
        cell[:NC, 2] = idx[0:-1, 0:-1].T.flatten()
        cell[NC:, 0] = idx[0:-1, 1:].T.flatten()
        cell[NC:, 1] = idx[0:-1, 0:-1].T.flatten()
        cell[NC:, 2] = idx[1:, 1:].T.flatten()

        if threshold is not None:
            bc = torch.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = torch.zeros(NN, dtype=torch.bool)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = torch.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        node.requires_grad_(require_grad)

        return cls(node, cell)

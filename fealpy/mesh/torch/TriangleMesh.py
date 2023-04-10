from typing import Optional

import torch
from torch import Tensor, device

from .Mesh2d import Mesh2d, Mesh2dDataStructure


class TriangleMeshDataStructure(Mesh2dDataStructure):

    localEdge = torch.tensor([(1, 2), (2, 0), (0, 1)])
    localFace = torch.tensor([(1, 2), (2, 0), (0, 1)])
    ccw = torch.tensor([0, 1, 2])

    NVC = 3
    NVE = 2
    NVF = 2

    NEC = 3
    NFC = 3

    localCell = torch.tensor([
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1)])

    def __init__(self, NN: int, cell: Tensor):
        super().__init__(NN=NN, cell=cell)


class TriangleMesh(Mesh2d):
    def __init__(self, node: Tensor, cell: Tensor, itype=torch.uint8, ftype=torch.float64):
        assert cell.shape[-1] == 3
        self.itype = itype
        self.ftype = ftype
        self.node = node.to(ftype)
        self.ds = TriangleMeshDataStructure(NN=node.shape[0], cell=cell.to(itype))
        self.device = node.device

    def geo_dimension(self) -> int:
        return self.node.shape[-1]

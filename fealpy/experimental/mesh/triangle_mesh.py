from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import SimplexMesh


class TriangleMesh(SimplexMesh):
    def __init__(self, node: TensorLike, cell: TensorLike) -> None:
        """
        """
        super().__init__(TD=2)
        kwargs = {'dtype': cell.dtype}
        self.cell = cell
        self.localEdge = bm.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.localFace = bm.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.ccw = bm.tensor([0, 1, 2], **kwargs)

        self.localCell = bm.tensor([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

        self.construct()

        self.node = node
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

    @classmethod
    def from_one_triangle(cls, meshtype='iso'):
        if meshtype == 'equ':
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3) / 2]], dtype=bm.float64)
        elif meshtype == 'iso':
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]], dtype=np.float64)
        cell = bm.tensor([[0, 1, 2]], dtype=bm.int_)
        return cls(node, cell)

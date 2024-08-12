from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import SimplexMesh, estr2dim
from .plot import Plotable


class PolygonMesh(Mesh, Plotable):
    def __init__(self, node: TensorLike, 
                 cell: Sequence[TensorLike, Optional[TensorLike]]) -> None:
        """
        """
        super().__init__(TD=2, itype=cell[0].dtype, ftype=node.dtype)
        kwargs = bm.context(cell[0]) 
        self.node = node
        if cell[1] is None: 
            assert cell[0].ndim = 2
            NC = cell[0].shape[0]
            NV = cell[0].shape[1]
            self.cell = (cell[0].reshape(-1), bm.arange(0, (NC+1)*NV, NV),
                         **kwargs)
        else:
            self.cell = cell

        self.meshtype = 'polygon'

        self.construct()

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

    def total_edge(self) -> TensorLike:
        cell, cellLocation = self.cell
        kwargs = bm.context(cell)
        totalEdge = bm.zeros((len(cell), 2), **kwargs)
        totalEdge[:, 0] = cell
        totalEdge[:-1, 1] = cell[1:]
        totalEdge[cellLocation[1:] - 1, 1] = cell[cellLocation[:-1]]
        return totalEdge

    total_face = total_edge

    def construct(self):
        """
        """
        cell, cellLocation = self.cell
        kwargs = bm.context(cell)

        totalEdge = self.total_edge()
        _, i0, j = bm.unique_all(bm.sort(totalEdge, axis=1), axis=0)

        NE = i0.shape[0]
        self.edge2cell = bm.zeros((NE, 4), **kwargs)

        i1 = np.zeros(NE, **kwargs)
        b = bm.arange(len(self._cell), **kwargs)
        bm.scatter(i1, j, b) 
        self.edge = totalEdge[i0]

        NV = self.number_of_vertices_of_cells()
        NC = self.number_of_cells()

        cellIdx = bm.repeat(range(NC), NV, **kwargs)
        localIdx = ranges(NV)

        self.edge2cell[:, 0] = cellIdx[i0]
        self.edge2cell[:, 1] = cellIdx[i1]
        self.edge2cell[:, 2] = localIdx[i0]
        self.edge2cell[:, 3] = localIdx[i1]
        self.cell2edge = j


PolygonMesh.set_ploter('polygon2d')

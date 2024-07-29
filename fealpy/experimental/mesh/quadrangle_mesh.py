from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .. import logger
from .utils import estr2dim

from .mesh_base import TensorMesh


class QuadrangleMesh(TensorMesh):
    def __init__(self, node, cell):
        """
        """
        super().__init__(TD=2)
        kwargs = {'dtype': cell.dtype}
        self.node = node
        self.cell = cell
        self.localEdge = bm.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.localFace = bm.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.ccw = bm.tensor([0, 1, 2, 3], **kwargs)

        self.localCell = None

        self.construct()

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

    def cell_area(self, index: Optional[Index]=None) -> TensorLike:
        """
        @brief 根据散度定理计算多边形的面积
        @note 请注意下面的计算方式不方便实现部分单元面积的计算
        """
        GD = self.GD
        if GD == 2:
            NC = self.number_of_cells()
            node = self.entity('node')
            edge = self.entity('edge')
            edge2cell = self.edge2cell

            t = self.edge_tangent()
            val = t[:, 1] * node[edge[:, 0], 0] - t[:, 0] * node[edge[:, 0], 1]

            a = bm.zeros(NC, dtype=self.ftype)
            bm.add.at(a, edge2cell[:, 0], val)

            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            bm.add.at(a, edge2cell[isInEdge, 1], -val[isInEdge])

            a /= 2.0

            return a[index]
        elif GD == 3:
            node = self.entity('node')
            cell = self.entity('cell')[index]

            v0 = node[cell[:, 1]] - node[cell[:, 0]]
            v1 = node[cell[:, 2]] - node[cell[:, 0]]
            v2 = node[cell[:, 3]] - node[cell[:, 0]]

            s1 = 0.5*bm.linalg.norm(bm.cross(v0, v1), axis=-1)
            s2 = 0.5*bm.linalg.norm(bm.cross(v1, v2), axis=-1)
            s = s1 + s2
            return s
    def entity_measure(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
        node = self.node

        if isinstance(etype, str):
            etype = estr2dim(self, etype)

        if etype == 0:
            return bm.tensor([0,], dtype=self.ftype)
        elif etype == 1:
            edge = self.entity(1, index)
            return bm.edge_length(edge, node)
        elif etype == 2:
            return self.cell_area(index=index)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")


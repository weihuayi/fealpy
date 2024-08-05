from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import HomogeneousMesh, estr2dim
from .triangle_mesh import TriangleMesh

class LagrangeTriangleMesh(HomogeneousMesh):
    def __init__(self, node: TensorLike, cell: TensorLike, p=1, surface=None,
            construct=False):
        super().__init__(TD=2)

        kwargs = bm.context(cell)
        self.p = p
        self.node = node
        self.cell = cell
        self.surface = surface

        self.localEdge = self.generate_local_lagrange_edges(p)
        self.localFace = self.localEdge
        self.ccw  = np.array([0, 1, 2], **kwargs)

        self.localCell = np.array([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

        if construct:
            self.construct()

        self.meshtype = 'ltri'

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}

    def construct(self):
        pass

    def generate_local_lagrange_edges(self, p: int) -> TensorLike:
        """
        Generate the local edges for Lagrange elements of order p.
        """
        TD = self.top_dimension()
        multiIndex = F.multi_index_matrix(p, TD)

        localEdge = np.zeros((3, p+1), dtype=np.int_)
        localEdge[2, :], = np.where(multiIndex[:, 2] == 0)
        localEdge[1, -1::-1], = np.where(multiIndex[:, 1] == 0)
        localEdge[0, :],  = np.where(multiIndex[:, 0] == 0)

        return localEdge

     # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell'):
        from .quadrature import TriangleQuadrature
        from .quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 2:
            quad = TriangleQuadrature(q)
        elif etype == 1:
            quad = GaussLegendreQuadrature(q)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad
 
    @classmethod
    def from_triangle_mesh(cls, mesh, p: int, surface=None):
        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if surface is not None:
            node, _ = surface.project(node)

        lmesh = cls(node, cell, p=p, construct=False)

        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell_to_edge()
        lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh 

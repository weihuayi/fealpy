from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import estr2dim
from .mesh_base import TensorMesh
from .quadrangle_mesh import QuadrangleMesh

class LagrangeQuadrangleMesh(TensorMesh):
    def __init__(self, node: TensorLike, cell: TensorLike, p=1, surface=None,
            construct=False):
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)

        kwargs = bm.context(cell)
        self.p = p
        self.node = node
        self.cell = cell
        self.surface = surface

        self.localEdge = self.generate_local_lagrange_edges(p)
        self.localFace = self.localEdge
        self.ccw = bm.tensor([0, 1, 2, 3], **kwargs)
        self.localCell = None

        if construct:
            self.construct()

        self.meshtype = 'lquad'


    def generate_local_lagrange_edges(self, p: int) -> TensorLike:
        """
        Generate the local edges for Lagrange elements of order p.
        """
        TD = self.top_dimension()
        multiIndex = bm.multi_index_matrix(p, TD)

        localEdge = bm.zeros((4, p+1), dtype=bm.int32)
        localEdge[3, :], = bm.where(multiIndex[:, 2] == 0)
        localEdge[2,:] = bm.flip(bm.where(multiIndex[:, 1] == 0)[1])
        localEdge[1,:] = bm.flip(bm.where(multiIndex[:, 1] == 0)[0])
        localEdge[0, :],  = bm.where(multiIndex[:, 0] == 0)

        return localEdge

    def reference_cell_measure(self):
        return 1
    

    @classmethod
    def from_quadrangle_mesh(cls, mesh, p: int, surface=None):
        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if surface is not None:
            node, _ = surface.project(node)

        lmesh = cls(node, cell, p=p, construct=True)

        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell_to_edge()
        #lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh 
     
    def quadrature_formula(self, q, etype: Union[int, str] = 'cell'):
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)
        if etype == 2:
            return TensorProductQuadrature((qf, qf))
        elif etype == 1:
            return qf
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

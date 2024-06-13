
from typing import Optional, Union, List

import numpy as np
import torch
from torch import Tensor

from fealpy.torch.mesh.mesh_base import _S
from fealpy.torch.mesh.quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import MeshDS, SimplexMesh, entity_str2dim

Index = Union[Tensor, int, slice]
_dtype = torch.dtype
_device = torch.device

_S = slice(None)


class IntervalMeshDataStructure(MeshDS):
    def __init__(self, NN: int, cell: Tensor) -> None:
        super().__init__(NN, 1)
        self.cell = cell

        self.construct()

    def total_face(self):
        return self.cell.reshape(-1, 1)


class IntervalMesh(SimplexMesh):
    def __init__(self, node: Tensor, cell: Tensor):
        if node.ndim == 1:
            node = node.reshape(-1, 1)
        self.node = node
        self.ds = IntervalMeshDataStructure(node.shape[0], cell)

    # entity
    def entity_measure(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        node = self.node
        if isinstance(etype, str):
            etype = entity_str2dim(self.ds, etype)
        if etype == 0:
            return torch.zeros((1,), dtype=node.dtype, device=node.device)
        elif etype == 1:
            edge = self.entity(1, index)
            return F.edge_length(node[edge])
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

    # integrator
    def integrator(self, q: int, etype: Union[int, str]='cell',
                   qtype: str='legendre') -> Quadrature: # TODO: other qtype
        from .quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = entity_str2dim(self.ds, etype)
        kwargs = {'dtype': self.ftype, 'device': self.device}
        if etype == 1:
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = entity_str2dim(self.ds, iptype)
        return F.simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        return F.simplex_gdof(p, self)

    def interpolation_points(self, p: int, index: Index=_S) -> Tensor:
        """Fetch all p-order interpolation points on the interval mesh."""
        node = self.entity('node')
        if p == 1:
            return node
        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")

        ipoint_list = []
        kwargs = {'dtype': self.ftype, 'device': self.device}

        GD = self.geo_dimension()
        ipoint_list.append(node)

        edge = self.entity('edge')
        w = torch.zeros((p - 1, 2), **kwargs)
        w[:, 0] = torch.arange(p - 1, 0, -1).div_(p)
        w[:, 1] = w[:, 0].flip(0)
        ipoints_from_edge = np.einsum('ij, kj... -> ki...', w,
                                      node[edge]).reshape(-1, GD)
        ipoint_list.append(ipoints_from_edge)

        return torch.cat(ipoint_list, dim=0)

    def cell_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        return self.edge_to_ipoint(p, index)

    def face_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        NN = self.number_of_nodes()
        return torch.arange(NN, dtype=self.ds.itype, device=self.device)

    # shape function
    def grad_lambda(self, index: Index=_S):
        return F.int_grad_lambda(self.node[self.ds.cell[index]])

    # constructor
    @classmethod
    def from_mesh_boundary(cls, mesh, /):
        """Construct a interval mesh from the boundary of a mesh with topology
        dimension 2.

        Args:
            mesh (Mesh): The mesh whose boundary is to construct the interval mesh on.

        Raises:
            ValueError: If the top-dimension of the mesh is not 2.

        Returns:
            IntervalMesh: _description_
        """
        if mesh.top_dimension() != 2:
            raise ValueError("The top-dimension of the mesh must be 2.")

        itype = mesh.ds.itype
        device = mesh.device
        is_bd_node = mesh.ds.boundary_node_flag()
        is_bd_face = mesh.ds.boundary_face_flag()
        node = mesh.entity('node', index=is_bd_node)
        face = mesh.entity('face', index=is_bd_face)
        NN = mesh.number_of_nodes()
        NN_bd = node.shape[0]

        I = torch.zeros((NN, ), dtype=itype, device=device)
        I[is_bd_node] = torch.arange(NN_bd, dtype=itype, device=device)
        face2bdnode = I[face]

        return cls(node=node, cell=face2bdnode)

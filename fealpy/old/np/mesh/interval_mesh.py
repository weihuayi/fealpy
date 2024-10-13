
from typing import Optional, Union, List

import numpy as np
from numpy.typing import NDArray

from fealpy.torch.mesh.mesh_base import _S
from fealpy.torch.mesh.quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import SimplexMesh, estr2dim

Index = Union[NDArray, int, slice]
_dtype = np.dtype

_S = slice(None)


class IntervalMesh(SimplexMesh):
    def __init__(self, node: NDArray, cell: NDArray):
        super().__init__(TD=1)
        self.cell = cell

        self.construct()

        if node.ndim == 1:
            node = node.reshape(-1, 1)
        self.node = node

    def total_face(self):
        return self.cell.reshape(-1, 1)

    # entity
    def entity_measure(self, etype: Union[int, str], index: Optional[Index]=None) -> NDArray:
        node = self.node
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            return np.zeros((1,), dtype=node.dtype)
        elif etype == 1:
            edge = self.entity(1, index)
            return F.edge_length(edge, node)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell',
                           qtype: str='legendre') -> Quadrature: # TODO: other qtype
        from .quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype}
        if etype == 1:
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    # ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> NDArray:
        """Fetch all p-order interpolation points on the interval mesh."""
        node = self.entity('node')
        if p == 1:
            return node
        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")

        ipoint_list = []
        kwargs = {'dtype': self.ftype}

        GD = self.geo_dimension()
        ipoint_list.append(node)

        edge = self.entity('edge')
        w = np.zeros((p - 1, 2), **kwargs)
        w[:, 0] = np.arange(p - 1, 0, -1)/p
        w[:, 1] = np.flip(w[:, 0], axis=0)
        ipoints_from_edge = np.einsum('ij, kj... -> ki...', w,
                                      node[edge]).reshape(-1, GD)
        ipoint_list.append(ipoints_from_edge)

        return np.concatenate(ipoint_list, axis=0)

    def cell_to_ipoint(self, p: int, index: Index=_S) -> NDArray:
        return self.edge_to_ipoint(p, index)

    def face_to_ipoint(self, p: int, index: Index=_S) -> NDArray:
        NN = self.number_of_nodes()
        return np.arange(NN, dtype=self.itype)

    # shape function
    def grad_lambda(self, index: Index=_S):
        return F.int_grad_lambda(self.cell[index], self.node)

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

        itype = mesh.itype
        is_bd_node = mesh.boundary_node_flag()
        is_bd_face = mesh.boundary_face_flag()
        node = mesh.entity('node', index=is_bd_node)
        face = mesh.entity('face', index=is_bd_face)
        NN = mesh.number_of_nodes()
        NN_bd = node.shape[0]

        I = np.zeros((NN, ), dtype=itype)
        I[is_bd_node] = np.arange(NN_bd, dtype=itype)
        face2bdnode = I[face]

        return cls(node=node, cell=face2bdnode)

from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor, device

from .mesh_data_structure import Mesh2dDataStructure
from .mesh import Mesh2d


class TriangleMeshDataStructure(Mesh2dDataStructure):
    # Constants Only
    localEdge = torch.tensor([(1, 2), (2, 0), (0, 1)])
    localFace = torch.tensor([(1, 2), (2, 0), (0, 1)])
    localCell = torch.tensor([
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1)])
    ccw = torch.tensor([0, 1, 2])

    NVC = 3
    NVE = 2
    NEC = 3


class TriangleMesh(Mesh2d):
    def __init__(self, node: Tensor, cell: Tensor):
        assert cell.shape[-1] == 3

        self.itype = cell.dtype
        self.ftype = node.dtype
        self.node = node
        self.ds = TriangleMeshDataStructure(NN=node.shape[0], cell=cell)
        self.device = node.device

    def uniform_refine(self, n: int=1):
        pass

    def integrator(self, k: int, etype: Union[int, str]):
        from ...quadrature import TriangleQuadrature, GaussLegendreQuadrature

        if etype in {'cell', 2}:
            return TriangleQuadrature(index=k)

        elif etype in {'edge', 'face', 1}:
            return GaussLegendreQuadrature(k)

        else:
            raise ValueError(f"Invalid entity type '{etype}'.")

    def shape_function(self, bc: Tensor, p: int=1) -> Tensor:
        multi_idx = self.multi_index_matrix(p=p, device=self.device)

        shape = bc.shape[:-1] + (1, 3)
        a_zero = torch.ones(shape, dtype=self.ftype, device=self.device)

        t = torch.arange(0, p, dtype=self.ftype, device=self.device)
        a_after_one = p * bc[..., None, :] - t.reshape(-1, 1)
        A_ = torch.cumprod(a_after_one, dim=-2)
        A = torch.cat([a_zero, A_], dim=-2)

        c = torch.arange(1, p+1, dtype=torch.int, device=self.device)
        P = 1.0/torch.cumprod(c, dim=0)
        A[..., 1:, :] *= P.reshape(-1, 1)

        idx = torch.arange(3, device=self.device)
        phi = torch.prod(A[..., multi_idx, idx], dim=-1)
        return phi

    def grad_shape_function(self, bc: Tensor, p: int, index=...):
        """
        @brief
        """
        pass

    @staticmethod
    def multi_index_matrix(p: int, etype: Union[int, str]=2, device: Optional[device]=None):
        """
        @brief Get p-order multi-index matrix in a triangle.

        @param[in] p: Positive integer.

        @return: Tensor with shape (ldof, 3).
        """
        if p < 0:
            raise ValueError(f"Order of multiple indexes should not be negative.")

        if etype in {'cell', 2}:
            ldof = (p+1)*(p+2)//2
            idx = torch.arange(0, ldof)
            idx0 = torch.floor((-1 + torch.sqrt(1+8*idx))/2)
            multi_idx = torch.zeros((ldof, 3), dtype=torch.int, device=device)
            multi_idx[:, 2] = idx - idx0*(idx0 + 1)/2
            multi_idx[:, 1] = idx0 - multi_idx[:, 2]
            multi_idx[:, 0] = p - multi_idx[:, 1] - multi_idx[:, 2]
            return multi_idx

        elif etype in {'face', 'edge', 1}:
            ldof = p + 1
            multi_idx = torch.zeros((ldof, 2), dtype=torch.int, device=device)
            multi_idx[:, 0] = torch.arange(p, -1, -1)
            multi_idx[:, 1] = p - multi_idx[:, 0]
            return multi_idx

        raise ValueError(f"Invalid entity type '{etype}'.")

    def number_of_local_ipoints(self, p: int, iptype: Union[int, str] = 'cell') -> int:
        if iptype in {'cell', 2}:
            return (p+1)*(p+2)//2
        elif iptype in {'face', 'edge', 1}:
            return p + 1
        elif iptype in {'node', 0}:
            return 1
        raise ValueError(f"Invalid entity type '{iptype}'.")

    def number_of_global_ipoints(self, p: int) -> int:
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        return NN + (p-1)*NE + (p-2)*(p-1)//2*NC

    def interpolation_points(self, p: int):
        node = self.entity('node')

        if p == 1:
            return node

        if p >= 2:
            GD = self.geo_dimension()
            edge = self.entity('edge')

            w = torch.zeros((p-1, 2), dtype=self.ftype)
            w[:, 0] = torch.arange(p-1, 0, -1)/p
            w[:, 1] = torch.arange(1, p, 1)/p
            ip_1 = torch.einsum('ij, ...jm -> ...im', w, node[edge, :]).reshape(-1, GD)

            if p >= 3:
                cell = self.entity('cell')
                multi_index = self.multi_index_matrix(p=p, etype='cell', device=self.device)
                is_edge = (multi_index == 0)
                is_in_cell = torch.all(~is_edge, dim=-1)
                w = multi_index[is_in_cell, :]/p
                ip_2 = torch.einsum('ij, kj... -> ki...', w, node[cell, :]).reshape(-1, GD)
                return torch.cat([node, ip_1, ip_2], dim=0)

            return torch.cat([node, ip_1])
        else:
            raise ValueError(f"Order of interpolation points must be\
                             an positive integer.")

    def cell_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    def edge_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    def face_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    def node_to_ipoint(self, p: int, index=np.s_[:]):
        pass

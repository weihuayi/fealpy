from typing import Optional, Union, List

import torch
from torch import Tensor

from fealpy.torch.mesh.mesh_base import _S
from fealpy.torch.mesh.quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import SimplexMesh, estr2dim

Index = Union[Tensor, int, slice]
_dtype = torch.dtype
_device = torch.device

_S = slice(None)


class TetrahedronMesh(SimplexMesh):
    def __init__(self, node: Tensor, cell: Tensor) -> None:
        super().__init__(TD=3)
        # constant tensors
        kwargs = {'dtype': cell.dtype, 'device': cell.device}
        self.cell = cell
        self.localEdge = torch.tensor([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)], **kwargs)
        self.localFace = torch.tensor([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], **kwargs)
        self.ccw = torch.tensor([0, 1, 2, 3], **kwargs)

        self.localCell = torch.tensor([
            (0, 1, 2, 3),
            (1, 2, 0, 3),
            (2, 0, 1, 3),
            (0, 1, 3, 2)], **kwargs)

        self.construct()

        self.node = node
        self._attach_functionals()
        self.nodedata = {}
        self.celldata = {}

    def _attach_functionals(self):
        GD = self.geo_dimension()
        if GD == 3:
            self._cell_volume = F.simplex_measure
            self._grad_lambda = F.tet_grad_lambda_3d
        else:
            logger.warn(f"{GD}D tetrahedron mesh is not well supported: "
                        "cell_volume and grad_lambda are not available. "
                        "Any operation involving them will fail.")
            
    # entity
    def entity_measure(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        node = self.node
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            return torch.tensor([0,], dtype=self.ftype, device=self.device)
        elif etype == 1:
            edge = self.entity(1, index)
            return F.edge_length(edge, node)
        elif etype == 2:
            face = self.entity(2, index)
            return self._face_area(face, node)
        elif etype == 3:
            cell = self.entity(3, index)
            return self._cell_volume(cell, node)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
        
    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell',
                           qtype: str='legendre') -> Quadrature: # TODO: other qtype
        from .quadrature import TetrahedronQuadrature
        from .quadrature import TriangleQuadrature
        from .quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype, 'device': self.device}
        if etype == 3:
            quad = TetrahedronQuadrature(q, **kwargs)
        elif etype == 2:
            quad = TriangleQuadrature(q, **kwargs)
        elif etype == 1:
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad
    
    # shape function
    def grad_lambda(self, index: Index=_S):
        return self._grad_lambda(self.cell[index], self.node, localFace=self.localFace)

    @classmethod
    def from_box(cls, box=[0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10, threshold=None, *,
                 itype: Optional[_dtype]=torch.int,
                 ftype: Optional[_dtype]=torch.float64,
                 device: Union[_device, str, None]=None,
                 require_grad: bool=False):
        """
        Generate a tetrahedral mesh for a box domain.

        Parameters:
            box (List[int]): 6 integers, the left, right, bottom, top, front, back of the box.
            nx (int, optional): Number of divisions along the x-axis, defaults to 10.
            ny (int, optional): Number of divisions along the y-axis, defaults to 10.
            nz (int, optional): Number of divisions along the z-axis, defaults to 10.
            threshold (Callable | None, optional): Optional function to filter cells.
                Based on their barycenter coordinates, defaults to None.

        Returns:
            TetrahedronMesh: Tetrahedral mesh instance.
        """
        fkwargs = {'dtype': ftype, 'device': device}
        ikwargs = {'dtype': itype, 'device': device}
        NN = (nx + 1) * (ny + 1) * (nz + 1)
        NC = nx * ny * nz
        X, Y, Z = torch.meshgrid(
            torch.linspace(box[0], box[1], nx + 1, **fkwargs),
            torch.linspace(box[2], box[3], ny + 1, **fkwargs),
            torch.linspace(box[4], box[5], nz + 1, **fkwargs),
            indexing='ij'
        )
        node = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1)

        idx = torch.arange(NN, **ikwargs).reshape(nx + 1, ny + 1, nz + 1)
        c = idx[:-1, :-1, :-1]

        cell = torch.zeros((NC, 8), **ikwargs)
        nyz = (ny + 1) * (nz + 1)
        cell[:, 0] = c.flatten()
        cell[:, 1] = cell[:, 0] + nyz
        cell[:, 2] = cell[:, 1] + nz + 1
        cell[:, 3] = cell[:, 0] + nz + 1
        cell[:, 4] = cell[:, 0] + 1
        cell[:, 5] = cell[:, 4] + nyz
        cell[:, 6] = cell[:, 5] + nz + 1
        cell[:, 7] = cell[:, 4] + nz + 1

        localCell = torch.tensor([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=torch.int32)
        cell = cell[:, localCell].reshape(-1, 4)

        if threshold is not None:
            bc = torch.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = torch.zeros(NN, dtype=torch.bool, device=device)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = torch.zeros(NN, dtype=cell.dtype, device=device)
            idxMap[isValidNode] = torch.arange(isValidNode.sum(), dtype=cell.dtype, device=device)
            cell = idxMap[cell]

        node.requires_grad_(require_grad)

        return cls(node, cell)

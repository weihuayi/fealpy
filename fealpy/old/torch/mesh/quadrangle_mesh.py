from typing import Optional, Union, List

import numpy as np
import torch
from torch import Tensor

from .quadrature import Quadrature

from .. import logger
from . import functional as F #TODO: maybe just import nessesary functioals
from .mesh_base import TensorMesh, estr2dim

Index = Union[Tensor, int, slice]
_dtype = torch.dtype
_device = torch.device
_S = slice(None)


class QuadrangleMesh(TensorMesh):
    def __init__(self, node: Tensor, cell: Tensor) -> None:
        super().__init__(TD=2)
        # constant tensors
        kwargs = {'dtype': cell.dtype, 'device': cell.device}
        self.cell = cell
        self.localEdge = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.localFace = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.ccw = torch.tensor([0, 1, 2, 3], **kwargs)

        self.localCell = torch.tensor([
            (0, 1, 2, 3),
            (1, 2, 3, 0),
            (2, 3, 0, 1),
            (3, 0, 1, 2)], **kwargs)

        self.construct()

        self.node = node
        GD = node.size(-1)

        if GD == 2: # TODO: implement functionals here.
            self._cell_area = F.tensor_measure
            self._grad_lambda = F.quad_grad_lambda_2d
        elif GD == 3:
            self._cell_area = F.tri_area_3d
            self._grad_lambda = F.tri_grad_lambda_3d
        else:
            logger.warn(f"{GD}D quadrangle mesh is not well supported: "
                        "cell_area and grad_lambda are not available. "
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
            cell = self.entity(2, index)
            return self._cell_area(cell, node)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
        
    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell',
                           qtype: str='legendre') -> Quadrature: # TODO: other qtype
        from .quadrature import QuadrangleQuadrature
        from .quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype, 'device': self.device}
        if etype == 2:
            quad = QuadrangleQuadrature(q, **kwargs)
        elif etype == 1:
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    def cell_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        cell = self.cell
        kwargs = {'dtype': self.itype, 'device': self.device}
        if p == 0:
            return torch.arange(len(cell), **kwargs).reshape((-1, 1))[index]
        if p == 1:
            return cell[index, [0, 3, 1, 2]] # Sort by y direction first, then by x direction

        face2cell = self.face_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        cell2ipoint = torch.zeros((NC, (p+1)*(p+1)), **kwargs)
        c2p = cell2ipoint.reshape((NC, p+1, p+1))

        e2p = self.edge_to_ipoint(p)
        flag = face2cell[:, 2] == 0
        c2p[face2cell[flag, 0], :, 0] = e2p[flag]
        flag = face2cell[:, 2] == 1
        c2p[face2cell[flag, 0], -1, :] = e2p[flag]
        flag = face2cell[:, 2] == 2
        c2p[face2cell[flag, 0], :, -1] = torch.flip(e2p[flag], dims=[1])
        flag = face2cell[:, 2] == 3
        c2p[face2cell[flag, 0], 0, :] = torch.flip(e2p[flag], dims=[1])

        iflag = face2cell[:, 0] != face2cell[:, 1]
        flag = iflag & (face2cell[:, 3] == 0)
        c2p[face2cell[flag, 1], :, 0] = torch.flip(e2p[flag], dims=[1])
        flag = iflag & (face2cell[:, 3] == 1)
        c2p[face2cell[flag, 1], -1, :] = torch.flip(e2p[flag], dims=[1])
        flag = iflag & (face2cell[:, 3] == 2)
        c2p[face2cell[flag, 1], :, -1] = e2p[flag]
        flag = iflag & (face2cell[:, 3] == 3)
        c2p[face2cell[flag, 1], 0, :] = e2p[flag]

        c2p[:, 1:-1, 1:-1] = NN + NE*(p-1) + torch.arange(NC*(p-1)*(p-1), **kwargs).reshape(NC, p-1, p-1)

        return cell2ipoint[index]
    
    # shape function
    def grad_lambda(self, index: Index=_S):
        return self._grad_lambda(self.cell[index], self.node)
            
    # constructor
    @classmethod
    def from_box(cls, box: List[int]=[0, 1, 0, 1], nx=10, ny=10, threshold=None, *,
                 itype: Optional[_dtype]=torch.int,
                 ftype: Optional[_dtype]=torch.float64,
                 device: Union[_device, str, None]=None,
                 require_grad: bool=False):
        """Generate a uniform triangle mesh for a box domain.

        Parameters:
            box (List[int]): 4 integers, the left, right, bottom, top of the box.\n
            nx (int, optional): Number of divisions along the x-axis, defaults to 10.\n
            ny (int, optional): Number of divisions along the y-axis, defaults to 10.\n
            threshold (Callable | None, optional): Optional function to filter cells.
                Based on their barycenter coordinates, defaults to None.

        Returns:
            TriangleMesh: Triangle mesh instance.
        """
        fkwargs = {'dtype': ftype, 'device': device}
        ikwargs = {'dtype': itype, 'device': device}
        NN = (nx + 1) * (ny + 1)
        NC = nx * ny
        X, Y = torch.meshgrid(
            torch.linspace(box[0], box[1], nx + 1, **fkwargs),
            torch.linspace(box[2], box[3], ny + 1, **fkwargs),
            indexing='ij'
        )
        node = torch.stack([X.ravel(), Y.ravel()], dim=-1)

        idx = torch.arange(NN, **ikwargs).reshape(nx + 1, ny + 1)
        cell = torch.zeros((NC, 4), **ikwargs)
        cell[:, 0] = idx[0:-1, 0:-1].flatten()
        cell[:, 1] = idx[1:, 0:-1].flatten()
        cell[:, 2] = idx[1:, 1:].flatten()
        cell[:, 3] = idx[0:-1, 1:].flatten()

        if threshold is not None:
            bc = torch.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = torch.zeros(NN, dtype=torch.bool)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = torch.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        node.requires_grad_(require_grad)

        return cls(node, cell)

from typing import Optional, Union, List

import torch
from torch import Tensor

from fealpy.old.torch.mesh.mesh_base import _S
from fealpy.old.torch.mesh.quadrature import Quadrature

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
    
    def cell_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        TD = self.top_dimension()
        edof = p+1
        fdof = (p+1)*(p+2)//2
        ldof = (p+1)*(p+2)*(p+3)//6

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()
        kwargs = {'dtype': self.itype, 'device': self.device}

        face = self.face
        cell = self.cell
        cell2face = self.cell_to_face()

        cell2ipoint = torch.zeros((NC, ldof), **kwargs)

        face2ipoint = self.face_to_ipoint(p)
        m2 = self.multi_index_matrix(p, TD-1).T
        m3 = self.multi_index_matrix(p, TD).T
        isFaceIPoint = (m3 == 0)

        fidx = torch.argsort(face, axis=1) # 第 i 个全局面顶点做一个排序
        fidx = torch.argsort(fidx, axis=1)
        for i in range(4):
            idx = list(range(4))
            idx.remove(i)
            idxj = torch.argsort(cell[:, idx], axis=1) #  (NC, 3)

            idxi = fidx[cell2face[:, i]]

            order = idxj[torch.arange(NC).reshape(-1, 1), idxi] # (NC, 3)
            # order 满足条件: fi - fj[np.arange(NC)[:, None], idx] = 0

            mi = m2[order]  # (NC, 3, fdof)
            k = mi[:, 1] + mi[:, 2] # (NC, fdof)
            a = k*(k+1)//2 + mi[:, 2] # (NC, fdof)
            cell2ipoint[:, isFaceIPoint[i]] = face2ipoint[cell2face[:, [i]], a]

        if p > 3:
            base = NN + (p-1)*NE + (fdof - 3*p)*NF
            idof = ldof - 4 - 6*(p - 1) - 4*(fdof - 3*p)
            isInCellIPoint = ~(isFaceIPoint[0] | isFaceIPoint[1] | isFaceIPoint[2] | isFaceIPoint[3])
            cell2ipoint[:, isInCellIPoint] = base + torch.arange(NC*idof).reshape(NC, idof)

        return cell2ipoint
    
    def face_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        TD = self.top_dimension()
        fdof = (p + 1) * (p + 2) // 2
        kwargs = {'dtype': self.itype, 'device': self.device}

        edgeIdx = torch.zeros((2, p + 1), **kwargs)
        edgeIdx[0, :] = torch.arange(p + 1, **kwargs)
        edgeIdx[1, :] = torch.flip(edgeIdx[0], dims=[0])

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()

        face = self.face
        edge = self.edge
        face2edge = self.face_to_edge()
        edge2ipoint = self.edge_to_ipoint(p)
        face2ipoint = torch.zeros((NF, fdof), **kwargs)

        faceIdx = self.multi_index_matrix(p, TD - 1)
        isEdgeIPoint = (faceIdx == 0)

        fe = torch.tensor([1, 0, 0], **kwargs)
        for i in range(3):
            I = torch.ones(NF, **kwargs)
            sign = (face[:, fe[i]] == edge[face2edge[:, i], 0])
            I[sign] = 0
            face2ipoint[:, isEdgeIPoint[:, i]] = edge2ipoint[face2edge[:, i], edgeIdx[I]]

        if p > 2:
            base = NN + (p - 1) * NE
            isInFaceIPoint = ~(isEdgeIPoint[:, 0] | isEdgeIPoint[:, 1] | isEdgeIPoint[:, 2])
            fidof = fdof - 3 * p
            face2ipoint[:, isInFaceIPoint] = base + torch.arange(NF * fidof,**kwargs).reshape(NF, fidof)

        return face2ipoint[index]

    
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

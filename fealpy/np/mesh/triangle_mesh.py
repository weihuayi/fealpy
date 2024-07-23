from typing import Optional, Union, List

import numpy as np
from numpy.typing import NDArray

from fealpy.np.mesh.mesh_base import _S
from fealpy.np.mesh.quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import SimplexMesh, estr2dim

Index = Union[NDArray, int, slice]
_dtype = np.dtype

_S = slice(None)


class TriangleMesh(SimplexMesh):
    def __init__(self, node: NDArray, cell: NDArray) -> None:
        super().__init__(TD=2)
        # constant tensors
        kwargs = {'dtype': cell.dtype}
        self.cell = cell
        self.localEdge = np.array([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.localFace = np.array([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.ccw = np.array([0, 1, 2], **kwargs)

        self.localCell = np.array([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

        self.construct()

        self.node = node
        GD = node.shape[-1]

        if GD == 2:
            self._cell_area = F.simplex_measure
            self._grad_lambda = F.tri_grad_lambda_2d
        elif GD == 3:
            self._cell_area = F.tri_area_3d
            self._grad_lambda = F.tri_grad_lambda_3d
        else:
            logger.warn(f"{GD}D triangle mesh is not well supported: "
                        "cell_area and grad_lambda are not available. "
                        "Any operation involving them will fail.")

    # entity
    def entity_measure(self, etype: Union[int, str], index: Optional[Index]=None) -> NDArray:
        node = self.node
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            return np.array([0,], dtype=self.ftype)
        elif etype == 1:
            edge = self.entity(1, index)
            return F.edge_length(node[edge])
        elif etype == 2:
            cell = self.entity(2, index)
            return self._cell_area(node[cell])
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
        
    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell',
                           qtype: str='legendre') -> Quadrature: # TODO: other qtype
        from .quadrature import TriangleQuadrature
        from .quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype}
        if etype == 2:
            quad = TriangleQuadrature(q, **kwargs)
        elif etype == 1:
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    # # integrator
    # def integrator(self, q: int, etype: Union[int, str]='cell',
    #                qtype: str='legendre') -> Quadrature: # TODO: other qtype
    #     from .quadrature import TriangleQuadrature
    #     from .quadrature import GaussLegendreQuadrature

    #     if isinstance(etype, str):
    #         etype = estr2dim(self, etype)
    #     kwargs = {'dtype': self.ftype}
    #     if etype == 2:
    #         quad = TriangleQuadrature(q, **kwargs)
    #     elif etype == 1:
    #         quad = GaussLegendreQuadrature(q, **kwargs)
    #     else:
    #         raise ValueError(f"Unsupported entity or top-dimension: {etype}")

    #     return quad

    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return F.simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        return F.simplex_gdof(p, self)

    def interpolation_points(self, p: int, index: Index=_S) -> NDArray:
        """Fetch all p-order interpolation points on the triangle mesh."""
        node = self.entity('node')
        if p == 1:
            return node
        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")

        ipoint_list = []
        kwargs = {'dtype': self.ftype}

        GD = self.geo_dimension()
        ipoint_list.append(node) # ipoints[:NN, :]

        edge = self.entity('edge')
        w = np.zeros((p - 1, 2), **kwargs)
        w[:, 0] = np.arange(p - 1, 0, -1, **kwargs) / p
        w[:, 1] = np.flip(w[:, 0], axis=0)
        ipoints_from_edge = np.einsum('ij, ...jm->...im', w,
                                         node[edge, :]).reshape(-1, GD) # ipoints[NN:NN + (p - 1) * NE, :]
        ipoint_list.append(ipoints_from_edge)

        if p >= 3:
            TD = self.top_dimension()
            cell = self.entity('cell')
            multiIndex = self.multi_index_matrix(p, TD)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:, 0] | isEdgeIPoints[:, 1] |
                                isEdgeIPoints[:, 2])
            multiIndex = multiIndex[isInCellIPoints, :]
            w = multiIndex.astype(self.ftype) / p

            ipoints_from_cell = np.einsum('ij, kj...->ki...', w,
                                          node[cell, :]).reshape(-1, GD) # ipoints[NN + (p - 1) * NE:, :]
            ipoint_list.append(ipoints_from_cell)

        return np.concatenate(ipoint_list, axis=0)  # (gdof, GD)

    def cell_to_ipoint(self, p: int, index: Index=_S) -> NDArray:
        cell = self.cell
        if p == 1:
            return cell[index]

        mi = self.multi_index_matrix(p, 2)
        idx0, = np.nonzero(mi[:, 0] == 0)
        idx1, = np.nonzero(mi[:, 1] == 0)
        idx2, = np.nonzero(mi[:, 2] == 0)
        kwargs = {'dtype': self.itype}

        face2cell = self.face_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoints(p, 'cell')
        c2p = np.zeros((NC, ldof), **kwargs)

        flag = face2cell[:, 2] == 0
        c2p[face2cell[flag, 0][:, None], idx0] = e2p[flag]

        flag = face2cell[:, 2] == 1
        idx1_ = np.flip(idx1, axis=0)
        c2p[face2cell[flag, 0][:, None], idx1_] = e2p[flag]

        flag = face2cell[:, 2] == 2
        c2p[face2cell[flag, 0][:, None], idx2] = e2p[flag]

        iflag = face2cell[:, 0] != face2cell[:, 1]

        flag = iflag & (face2cell[:, 3] == 0)
        idx0_ = np.flip(idx0, axis=0)
        c2p[face2cell[flag, 1][:, None], idx0_] = e2p[flag]

        flag = iflag & (face2cell[:, 3] == 1)
        c2p[face2cell[flag, 1][:, None], idx1] = e2p[flag]

        flag = iflag & (face2cell[:, 3] == 2)
        idx2_ = np.flip(idx2, axis=0)
        c2p[face2cell[flag, 1][:, None], idx2_] = e2p[flag]

        cdof = (p-1)*(p-2)//2
        flag = np.sum(mi > 0, axis=1) == 3
        c2p[:, flag] = NN + NE*(p-1) + np.arange(NC*cdof, **kwargs).reshape(NC, cdof)
        return c2p[index]

    def face_to_ipoint(self, p: int, index: Index=_S) -> NDArray:
        return self.edge_to_ipoint(p, index)

    # shape function
    def grad_lambda(self, index: Index=_S):
        return self._grad_lambda(self.node[self.cell[index]])

    # constructor
    @classmethod
    def from_box(cls, box: List[int]=[0, 1, 0, 1], nx=10, ny=10, threshold=None, *,
                 itype: Optional[_dtype]=np.int_,
                 ftype: Optional[_dtype]=np.float64):
        """Generate a uniform triangle mesh for a box domain .

        Parameters:
            box (List[int]):

            nx (int): Number of divisions along the x-axis (default: 10)

            ny (int): Number of divisions along the y-axis (default: 10)

            threshold: Optional function to filter cells based on their barycenter coordinates (default: None)

        Returns:
            TriangleMesh: Triangle mesh instance.
        """
        fkwargs = {'dtype': ftype}
        ikwargs = {'dtype': itype}
        NN = (nx + 1) * (ny + 1)
        NC = nx * ny
        X, Y = np.meshgrid(
            np.linspace(box[0], box[1], nx + 1, **fkwargs),
            np.linspace(box[2], box[3], ny + 1, **fkwargs),
            indexing='ij'
        )
        node = np.stack([X.ravel(), Y.ravel()], axis=-1)

        idx = np.arange(NN, **ikwargs).reshape(nx + 1, ny + 1)
        cell = np.zeros((2 * NC, 3), **ikwargs)
        cell[:NC, 0] = idx[1:, 0:-1].T.flatten()
        cell[:NC, 1] = idx[1:, 1:].T.flatten()
        cell[:NC, 2] = idx[0:-1, 0:-1].T.flatten()
        cell[NC:, 0] = idx[0:-1, 1:].T.flatten()
        cell[NC:, 1] = idx[0:-1, 0:-1].T.flatten()
        cell[NC:, 2] = idx[1:, 1:].T.flatten()

        if threshold is not None:
            bc = np.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = np.zeros(NN, dtype=np.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = np.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        return cls(node, cell)


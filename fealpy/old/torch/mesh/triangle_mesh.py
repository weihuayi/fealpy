
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


class TriangleMesh(SimplexMesh):
    def __init__(self, node: Tensor, cell: Tensor) -> None:
        super().__init__(TD=2)
        # constant tensors
        kwargs = {'dtype': cell.dtype, 'device': cell.device}
        self.cell = cell
        self.localEdge = torch.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.localFace = torch.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.ccw = torch.tensor([0, 1, 2], **kwargs)

        self.localCell = torch.tensor([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

        self.construct()

        self.node = node
        self._attach_functionals()
        self.nodedata = {}
        self.celldata = {}

    def _attach_functionals(self):
        GD = self.geo_dimension()
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
        from .quadrature import TriangleQuadrature
        from .quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype, 'device': self.device}
        if etype == 2:
            quad = TriangleQuadrature(q, **kwargs)
        elif etype == 1:
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    # ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> Tensor:
        """Fetch all p-order interpolation points on the triangle mesh."""
        node = self.entity('node')
        if p == 1:
            return node
        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")

        ipoint_list = []
        kwargs = {'dtype': self.ftype, 'device': self.device}

        GD = self.geo_dimension()
        ipoint_list.append(node) # ipoints[:NN, :]

        edge = self.entity('edge')
        w = torch.zeros((p - 1, 2), **kwargs)
        w[:, 0] = torch.arange(p - 1, 0, -1, **kwargs).div_(p)
        w[:, 1] = w[:, 0].flip(0)
        ipoints_from_edge = torch.einsum('ij, ...jm->...im', w,
                                         node[edge, :]).reshape(-1, GD) # ipoints[NN:NN + (p - 1) * NE, :]
        ipoint_list.append(ipoints_from_edge)

        if p >= 3:
            TD = self.top_dimension()
            cell = self.entity('cell')
            multiIndex = self.multi_index_matrix(p, TD)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:, 0] | isEdgeIPoints[:, 1] |
                                isEdgeIPoints[:, 2])
            w = multiIndex[isInCellIPoints, :].to(self.ftype).div_(p)
            ipoints_from_cell = torch.einsum('ij, kj...->ki...', w,
                                          node[cell, :]).reshape(-1, GD) # ipoints[NN + (p - 1) * NE:, :]
            ipoint_list.append(ipoints_from_cell)

        return torch.cat(ipoint_list, dim=0)  # (gdof, GD)

    def cell_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        cell = self.cell
        if p == 1:
            return cell[index]

        mi = self.multi_index_matrix(p, 2)
        idx0, = torch.nonzero(mi[:, 0] == 0, as_tuple=True)
        idx1, = torch.nonzero(mi[:, 1] == 0, as_tuple=True)
        idx2, = torch.nonzero(mi[:, 2] == 0, as_tuple=True)
        kwargs = {'dtype': self.itype, 'device': self.device}

        face2cell = self.face_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoints(p, 'cell')
        c2p = torch.zeros((NC, ldof), **kwargs)

        flag = face2cell[:, 2] == 0
        c2p[face2cell[flag, 0][:, None], idx0] = e2p[flag]

        flag = face2cell[:, 2] == 1
        c2p[face2cell[flag, 0][:, None], idx1.flip(0)] = e2p[flag]

        flag = face2cell[:, 2] == 2
        c2p[face2cell[flag, 0][:, None], idx2] = e2p[flag]

        iflag = face2cell[:, 0] != face2cell[:, 1]

        flag = iflag & (face2cell[:, 3] == 0)
        c2p[face2cell[flag, 1][:, None], idx0.flip(0)] = e2p[flag]

        flag = iflag & (face2cell[:, 3] == 1)
        c2p[face2cell[flag, 1][:, None], idx1] = e2p[flag]

        flag = iflag & (face2cell[:, 3] == 2)
        c2p[face2cell[flag, 1][:, None], idx2.flip(0)] = e2p[flag]

        cdof = (p-1)*(p-2)//2
        flag = torch.sum(mi > 0, axis=1) == 3
        c2p[:, flag] = NN + NE*(p-1) + torch.arange(NC*cdof, **kwargs).reshape(NC, cdof)
        return c2p[index]

    def face_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        return self.edge_to_ipoint(p, index)

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
        cell = torch.zeros((2 * NC, 3), **ikwargs)
        cell[:NC, 0] = idx[1:, 0:-1].T.flatten()
        cell[:NC, 1] = idx[1:, 1:].T.flatten()
        cell[:NC, 2] = idx[0:-1, 0:-1].T.flatten()
        cell[NC:, 0] = idx[0:-1, 1:].T.flatten()
        cell[NC:, 1] = idx[0:-1, 0:-1].T.flatten()
        cell[NC:, 2] = idx[1:, 1:].T.flatten()

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

    @classmethod
    def from_numpy(cls, mesh):
        import numpy as np

        new_mesh = cls.__new__(cls)
        SimplexMesh.__init__(new_mesh, TD=2)

        for name, tensor_obj in mesh.__dict__.items():
            if isinstance(tensor_obj, np.ndarray):
                setattr(new_mesh, name, torch.from_numpy(tensor_obj))

        # NOTE: Meshes in old numpy version has `ds`` instead of `_entity_storage`.
        if hasattr(mesh, '_entity_storage'):
            for etype, entity in mesh._entity_storage.items():
                new_mesh._entity_storage[etype] = torch.from_numpy(entity)

        if hasattr(mesh, 'ds'):
            for name, tensor_obj in mesh.ds.__dict__.items():
                if isinstance(tensor_obj, np.ndarray):
                    setattr(new_mesh, name, torch.from_numpy(tensor_obj))

        new_mesh._attach_functionals()

        return new_mesh
    
    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_TRIANGLE = 5
            return VTK_TRIANGLE
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE
    
    def to_vtk(self, fname:Optional[str]=None, etype:Union[int, str]='cell', index:Index=_S):
        """
        @brief 把网格转化为 vtk 的数据格式
        """
        from .vtk_extent import write_to_vtu

        fkwargs = {'dtype': self.ftype, 'device': self.device}
        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = torch.concatenate((node, torch.zeros((node.shape[0], 1), **fkwargs)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        NV = cell.shape[-1]

        cell = torch.cat((torch.zeros((len(cell), 1), **fkwargs), cell), dim=1)
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                         nodedata=self.nodedata,
                         celldata=self.celldata)

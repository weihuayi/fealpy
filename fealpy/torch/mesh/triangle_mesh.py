
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from fealpy.torch.mesh.mesh_base import _S

from .. import logger
from . import functional as F
from . import mesh_kernel as K
from .mesh_base import HomoMeshDataStructure, HomoMesh

Index = Union[Tensor, int, slice]
_dtype = torch.dtype
_device = torch.device

_S = slice(None)


class TriangleMeshDataStructure(HomoMeshDataStructure):
    def __init__(self, NN: int, cell: Tensor):
        super().__init__(NN, 2, cell)
        # constant tensors
        kwargs = {'dtype': cell.dtype, 'device': cell.device}
        self.localEdge = torch.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.localFace = torch.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.ccw = torch.tensor([0, 1, 2], **kwargs)

        self.localCell = torch.tensor([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

    def total_face(self):
        return self.cell[..., self.localFace].reshape(-1, 2)

    # TODO: this is not correct. So, face2cell is unavailable.
    def construct(self) -> None:
        kwargs = {'dtype': self.itype, 'device': self.device}
        NC = self.cell.shape[0]
        NFC = self.cell.shape[1]

        totalFace = self.total_face()
        _, i0_np, j_np = np.unique(
            torch.sort(totalFace, dim=1)[0].cpu().numpy(),
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = totalFace[i0_np, :] # this also adds the edge in 2-d meshes
        NF = i0_np.shape[0]

        i1_np = np.zeros(NF, **kwargs)
        i1_np[j_np] = np.arange(NFC*NC, **kwargs)

        self.cell2edge = torch.from_numpy(j_np).to(self.device).reshape(NC, NFC)
        self.cell2face = self.cell2edge

        face2cell_np = np.stack([i0_np//NFC, i1_np//NFC, i0_np%NFC, i1_np%NFC], axis=-1)
        self.face2cell = torch.from_numpy(face2cell_np).to(self.device)
        self.edge2cell = self.face2cell

        logger.info(f"Mesh toplogy relation constructed, with {NF} edge (or face), "
                    f"on device {self.device}")


class TriangleMesh(HomoMesh):
    def __init__(self, node: Tensor, cell: Tensor) -> None:
        self.node = node
        self.ds = TriangleMeshDataStructure(node.shape[0], cell)

        GD = node.size(-1)

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

    def cell_to_ipoint(self, p: int, index: Index=None) -> Tensor:
        cell = self.ds.cell
        if p == 1:
            return cell[index]

        mi = self.multi_index_matrix(p, 2)
        idx0, = torch.nonzero(mi[:, 0] == 0)
        idx1, = torch.nonzero(mi[:, 1] == 0)
        idx2, = torch.nonzero(mi[:, 2] == 0)

        face2cell = self.ds.face_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoint()

    def grad_lambda(self, index: Index=_S):
        return self._grad_lambda(self.node[self.ds.cell[index]])

    def shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        TD = bc.shape[-1] - 1
        mi = mi or F.multi_index_matrix(p, TD, dtype=self.ds.itype, device=self.device)
        phi = K.simplex_shape_function(bc, p, mi)
        if variable == 'u':
            return phi
        elif variable == 'x':
            return phi.unsqueeze_(0)
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")

    def grad_shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        TD = bc.shape[-1] - 1
        mi = mi or F.multi_index_matrix(p, TD, dtype=self.ds.itype, device=self.device)
        R = K.simplex_grad_shape_function(bc, p, mi)
        if variable == 'u':
            return R
        elif variable == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = torch.einsum('...ij, kjm -> k...im', Dlambda, R)
            # NOTE: the subscript 'k': cell, 'i': dof, 'j': bc, 'm': dimension, '...': batch
            return gphi
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")

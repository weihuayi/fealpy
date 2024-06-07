from typing import Optional, Union, List

import numpy as np
import torch
from torch import Tensor

from .quadrature import Quadrature

from .. import logger
from . import functional as F #TODO: maybe just import nessesary functioals
from . import mesh_kernel as K
from .mesh_base import HomoMeshDataStructure, HomoMesh, entity_str2dim

Index = Union[Tensor, int, slice]
_dtype = torch.dtype
_device = torch.device
_S = slice(None)


class QuadrangleMeshDataStructure(HomoMeshDataStructure):
    def __init__(self, NN: int, cell: Tensor):
        super().__init__(NN, 2, cell)
        # constant tensors
        kwargs = {'dtype': cell.dtype, 'device': cell.device}
        self.localEdge = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.localFace = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.ccw = torch.tensor([0, 1, 2, 3], **kwargs)

        self.localCell = torch.tensor([
            (0, 1, 2, 3),
            (1, 2, 3, 0),
            (2, 3, 0, 1),
            (3, 0, 1, 2)], **kwargs)

        self.construct()


    def construct(self) -> None:
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

        i1_np = np.zeros(NF, dtype=i0_np.dtype)
        i1_np[j_np] = np.arange(NFC*NC, dtype=i0_np.dtype)

        self.cell2edge = torch.from_numpy(j_np).to(self.device).reshape(NC, NFC)
        self.cell2face = self.cell2edge

        face2cell_np = np.stack([i0_np//NFC, i1_np//NFC, i0_np%NFC, i1_np%NFC], axis=-1)
        self.face2cell = torch.from_numpy(face2cell_np).to(self.device)
        self.edge2cell = self.face2cell

        logger.info(f"Mesh toplogy relation constructed, with {NF} edge (or face), "
                    f"on device {self.device}")


class QuadrangleMesh(HomoMesh):
    ds: QuadrangleMeshDataStructure
    def __init__(self, node: Tensor, cell: Tensor) -> None:
        self.node = node
        self.ds = QuadrangleMeshDataStructure(node.shape[0], cell)

        GD = node.size(-1)

        if GD == 2:
            self._cell_area = F.simplex_measure
            self._grad_lambda = F.tri_grad_lambda_2d
        elif GD == 3:
            self._cell_area = F.tri_area_3d
            self._grad_lambda = F.tri_grad_lambda_3d
        else:
            logger.warn(f"{GD}D quadrangle mesh is not well supported: "
                        "cell_area and grad_lambda are not available. "
                        "Any operation involving them will fail.")

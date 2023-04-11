from typing import Optional, Union

import torch
from torch import Tensor, device

from .Mesh2d import Mesh2d, Mesh2dDataStructure


class TriangleMeshDataStructure(Mesh2dDataStructure):

    localEdge = torch.tensor([(1, 2), (2, 0), (0, 1)])
    localFace = torch.tensor([(1, 2), (2, 0), (0, 1)])
    localCell = torch.tensor([
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1)])
    ccw = torch.tensor([0, 1, 2])

    NVC = 3
    NVE = 2
    NVF = 2

    NEC = 3
    NFC = 3


class TriangleMesh(Mesh2d):
    def __init__(self, node: Tensor, cell: Tensor, itype=torch.uint8, ftype=torch.float64):
        assert cell.shape[-1] == 3
        self.itype = itype
        self.ftype = ftype
        self.node = node.to(ftype)
        self.ds = TriangleMeshDataStructure(NN=node.shape[0], cell=cell.to(itype))
        self.device = node.device

    def geo_dimension(self) -> int:
        return self.node.shape[-1]

    def uniform_refine(self):
        return super().uniform_refine()

    def shape_function(self, bc: Tensor, p: int=1) -> Tensor:
        """
        @brief
        """
        TD = bc.shape[-1] - 1
        multi_idx = self.multi_index_matrix(p=p, device=self.device)
        c = torch.arange(1, p+1, dtype=torch.int, device=self.device)
        P = 1.0/np.multiply.accumulate(c)
        t = torch.arange(0, p, dtype=torch.int, device=self.device)

        shape = bc.shape[:-1] + (p+1, TD+1)
        A = torch.ones(shape, dtype=self.ftype, device=self.device)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        idx = torch.arange(TD+1, device=self.device)
        phi = torch.prod(A[..., multi_idx, idx], dim=-1)
        return phi

    def grad_shape_function(self, bc: Tensor, index=...):
        """
        @brief
        """
        ...

    @staticmethod
    def multi_index_matrix(p: int, etype: Union[int, str]=2, device: Optional[device]=None):
        """
        @brief Get p-order multi-index matrix in triangle.

        @param[in] p: Positive integer.

        @return: Tensor with shape (ldof, 3).
        """
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

    def interpolation_points(self):
        return super().interpolation_points()
